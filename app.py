"""
Hospital Appointment No-Show Predictor & Overbooking Simulator
Streamlit frontend
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="No-Show Predictor & Overbooking Simulator",
    page_icon="🏥",
    layout="wide",
)

DATA_DIR = "data"

# --------------------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    best_model = joblib.load(f"{DATA_DIR}/best_model.pkl")
    calibrated_model = joblib.load(f"{DATA_DIR}/calibrated_model.pkl")
    return best_model, calibrated_model


@st.cache_data(show_spinner="Loading data...")
def load_raw_data():
    patients = pd.read_csv(f"{DATA_DIR}/patients.csv")
    appointments = pd.read_csv(f"{DATA_DIR}/appointments.csv")
    return patients, appointments


@st.cache_data(show_spinner=False)
def load_test_predictions():
    return pd.read_csv(f"{DATA_DIR}/test_predictions.csv")


@st.cache_data(show_spinner=False)
def load_policy_summary():
    summary = pd.read_csv(f"{DATA_DIR}/policy_comparison_summary.csv")
    daily = pd.read_csv(f"{DATA_DIR}/smart_policy_daily_results.csv")
    return summary, daily


@st.cache_data(show_spinner=False)
def build_merged(patients, appointments):
    df = appointments.merge(patients, on="patient_id", how="left", suffixes=("", "_pat"))
    df["scheduled_date"] = pd.to_datetime(df["scheduled_date"], errors="coerce")
    df["appointment_date"] = pd.to_datetime(df["appointment_date"], errors="coerce")
    df["lead_time_days"] = (df["appointment_date"] - df["scheduled_date"]).dt.days
    df["appointment_weekday"] = df["appointment_date"].dt.day_name()
    df["appointment_month"] = df["appointment_date"].dt.month_name()
    df["is_weekend"] = df["appointment_date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["no_show_flag"] = (df["no_show"] == "Yes").astype(int)
    return df


CATEGORICAL_COLS = [
    "clinic_id", "doctor_id", "specialty", "weather_condition", "payment_method",
    "gender", "neighborhood", "income_bracket", "insurance_type",
    "appointment_weekday", "appointment_month",
]

NUMERIC_DEFAULTS_SOURCE_COLS = [
    "sms_reminder_sent", "temperature_c", "distance_to_clinic_km", "waitlist_flag",
    "appointment_cost", "previous_appointments_count", "previous_noshows_count",
    "age", "has_hypertension", "has_diabetes", "has_alcoholism", "disability_level",
]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def encode_batch(df_raw: pd.DataFrame, feature_names) -> pd.DataFrame:
    """Recreate the exact one-hot encoding / column layout the models were trained on."""
    df = df_raw.copy()

    # Derive engineered date features if raw dates are present and engineered ones are not
    if "scheduled_date" in df.columns and "appointment_date" in df.columns:
        sched = pd.to_datetime(df["scheduled_date"], errors="coerce")
        appt = pd.to_datetime(df["appointment_date"], errors="coerce")
        if "lead_time_days" not in df.columns:
            df["lead_time_days"] = (appt - sched).dt.days
        if "appointment_weekday" not in df.columns:
            df["appointment_weekday"] = appt.dt.day_name()
        if "appointment_month" not in df.columns:
            df["appointment_month"] = appt.dt.month_name()
        if "is_weekend" not in df.columns:
            df["is_weekend"] = appt.dt.dayofweek.isin([5, 6]).astype(int)

    # Fill numeric NaNs with column median (fallback 0), categorical NaNs with "Unknown"
    for col in df.columns:
        if col in CATEGORICAL_COLS:
            df[col] = df[col].fillna("Unknown").astype(str)
        elif df[col].dtype.kind in "fi":
            med = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(med) else med)

    present_cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=present_cat_cols)

    # Align to the exact training feature layout (drops reference categories, adds missing ones as 0)
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)
    return df_aligned.astype(float)


def risk_bucket(p: float) -> tuple:
    if p < 0.20:
        return "Low", "🟢"
    elif p < 0.40:
        return "Moderate", "🟡"
    elif p < 0.60:
        return "High", "🟠"
    else:
        return "Very High", "🔴"


def simulate_policy(df, pool, capacity, revenue_per_patient, cost_per_overflow,
                     policy="none", fixed_overbook_pct=0.2, seed_base=0):
    n_days = len(df) // capacity
    results = []
    for day in range(n_days):
        day_patients = df.iloc[day * capacity: (day + 1) * capacity]

        if policy == "none":
            booked = day_patients
        elif policy == "fixed":
            n_extra = int(round(capacity * fixed_overbook_pct))
            extra = pool.sample(n=min(n_extra, len(pool)), random_state=seed_base + day)
            booked = pd.concat([day_patients, extra])
        else:  # smart
            expected_noshows = day_patients["predicted_noshow_proba"].sum()
            n_extra = int(round(expected_noshows))
            if n_extra > 0:
                extra = pool.sample(n=min(n_extra, len(pool)), random_state=seed_base + day)
                booked = pd.concat([day_patients, extra])
            else:
                booked = day_patients

        shows = (booked["actual_no_show"] == 0).sum()
        seen = min(shows, capacity)
        overflow = max(0, shows - capacity)
        empty_slots = max(0, capacity - shows)

        revenue = seen * revenue_per_patient
        overflow_cost = overflow * cost_per_overflow
        net_revenue = revenue - overflow_cost

        results.append({
            "day": day, "booked": len(booked), "shows": shows, "seen": seen,
            "overflow": overflow, "empty_slots": empty_slots,
            "utilization_pct": seen / capacity * 100,
            "overcrowded": int(overflow > 0),
            "revenue": revenue, "overflow_cost": overflow_cost, "net_revenue": net_revenue,
        })
    return pd.DataFrame(results)


def summarize(results, name):
    return {
        "Policy": name,
        "Avg Utilization %": round(results["utilization_pct"].mean(), 2),
        "Overcrowded Days %": round(results["overcrowded"].mean() * 100, 2),
        "Avg Empty Slots/Day": round(results["empty_slots"].mean(), 2),
        "Total Net Revenue": round(results["net_revenue"].sum(), 0),
    }


# --------------------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------------------
st.sidebar.title("🏥 No-Show Predictor")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Insights (EDA)", "🔮 Predict a No-Show",
     "📁 Batch Prediction", "🧮 Overbooking Simulator", "📈 Model Performance"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Predicts patient no-show risk and simulates overbooking policies "
    "to help clinics maximize utilization and revenue."
)

best_model, calibrated_model = load_models()
FEATURE_NAMES = list(calibrated_model.feature_names_in_)
patients_df, appointments_df = load_raw_data()

# --------------------------------------------------------------------------------------
# 1. OVERVIEW
# --------------------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("🏥 Hospital Appointment No-Show Predictor & Overbooking Simulator")
    st.markdown(
        "An end-to-end tool that predicts which patients are likely to miss their "
        "appointments, and uses those predictions to simulate a smarter overbooking "
        "policy — turning a classification model into a real business decision-support tool."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{len(patients_df):,}")
    c2.metric("Appointments", f"{len(appointments_df):,}")
    overall_rate = (appointments_df["no_show"] == "Yes").mean() * 100
    c3.metric("Overall No-Show Rate", f"{overall_rate:.1f}%")
    summary_df, _ = load_policy_summary()
    best_rev = summary_df.loc[summary_df["Total Net Revenue"].idxmax()]
    c4.metric("Best Policy", best_rev["Policy"], f"Rs. {best_rev['Total Net Revenue']:,.0f}")

    st.markdown("### Key Findings")
    st.markdown(
        """
- No-show rate is highest on **Monday** and lowest on **Thursday**
- Appointments with an **SMS reminder sent** have a notably lower no-show rate
- No-show rate **rises steadily with lead time** (longer wait → more no-shows)
- Rainy/stormy weather raises no-show rate compared to sunny days
- Patients with chronic conditions are **more reliable**, not less
- A **calibrated Random Forest** model, fed into an overbooking simulator, delivers
  the highest clinic utilization and net revenue of the three policies tested
        """
    )

    st.markdown("### Policy Comparison (from the notebook simulation)")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ About probability calibration (why two models?)"):
        st.markdown(
            """
`best_model.pkl` is a **class-balanced** Random Forest — good for classification
metrics (precision/recall) but its predicted probabilities are inflated for the
minority (no-show) class.

`calibrated_model.pkl` is a Random Forest trained **without** class balancing and
wrapped in `CalibratedClassifierCV`. Its probabilities closely match real-world
no-show rates, so it's the one used for the overbooking simulator and for the
risk scores shown in this app.
            """
        )

# --------------------------------------------------------------------------------------
# 2. EDA / INSIGHTS
# --------------------------------------------------------------------------------------
elif page == "📊 Insights (EDA)":
    st.title("📊 Exploratory Insights")
    merged = build_merged(patients_df, appointments_df)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Weekday & SMS", "Lead Time & Weather", "Chronic Conditions", "Correlation"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            wd = merged.groupby("appointment_weekday")["no_show_flag"].mean().reindex(WEEKDAYS) * 100
            fig = px.bar(wd, y=wd.values, x=wd.index, labels={"y": "No-Show Rate %", "x": "Weekday"},
                         title="No-Show Rate by Weekday", color=wd.values, color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sms = merged.groupby("sms_reminder_sent")["no_show_flag"].mean() * 100
            sms.index = sms.index.map({0.0: "No Reminder", 1.0: "Reminder Sent"})
            fig = px.bar(sms, y=sms.values, x=sms.index, labels={"y": "No-Show Rate %", "x": ""},
                         title="No-Show Rate: SMS Reminder Effect", color=sms.index)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            bins = [-1, 3, 7, 14, 30, 400]
            labels = ["0-3", "4-7", "8-14", "15-30", "30+"]
            merged["lead_bucket"] = pd.cut(merged["lead_time_days"], bins=bins, labels=labels)
            lt = merged.groupby("lead_bucket", observed=True)["no_show_flag"].mean() * 100
            fig = px.line(lt, y=lt.values, x=lt.index, markers=True,
                          labels={"y": "No-Show Rate %", "x": "Lead Time (days)"},
                          title="No-Show Rate by Lead Time")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            wc = merged.groupby("weather_condition")["no_show_flag"].mean().sort_values() * 100
            fig = px.bar(wc, y=wc.values, x=wc.index, labels={"y": "No-Show Rate %", "x": "Weather"},
                         title="No-Show Rate by Weather Condition", color=wc.values,
                         color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        merged["chronic_flag"] = merged["has_hypertension"].fillna(0) + merged["has_diabetes"].fillna(0)
        merged["chronic_group"] = merged["chronic_flag"].map({0: "None", 1: "One", 2: "Both"})
        cg = merged.groupby("chronic_group")["no_show_flag"].mean().reindex(["None", "One", "Both"]) * 100
        fig = px.bar(cg, y=cg.values, x=cg.index, labels={"y": "No-Show Rate %", "x": "Chronic Conditions"},
                     title="No-Show Rate by Chronic Condition Count", color=cg.values,
                     color_continuous_scale="Greens")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Patients managing chronic conditions tend to be more reliable attendees.")

    with tab4:
        numeric_cols = ["age", "lead_time_days", "distance_to_clinic_km", "temperature_c",
                         "previous_appointments_count", "previous_noshows_count",
                         "appointment_cost", "no_show_flag"]
        corr = merged[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------
# 3. SINGLE PREDICTION
# --------------------------------------------------------------------------------------
elif page == "🔮 Predict a No-Show":
    st.title("🔮 Predict No-Show Risk for a Single Appointment")
    st.caption("Fill in patient & appointment details to get a calibrated no-show probability.")

    with st.form("predict_form"):
        st.subheader("Patient")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 0, 110, 40)
        gender = c2.selectbox("Gender", sorted(patients_df["gender"].dropna().unique()))
        neighborhood = c3.selectbox("Neighborhood", sorted(patients_df["neighborhood"].dropna().unique()))

        c4, c5, c6 = st.columns(3)
        income_bracket = c4.selectbox("Income Bracket", sorted(patients_df["income_bracket"].dropna().unique()))
        insurance_type = c5.selectbox("Insurance Type", sorted(patients_df["insurance_type"].dropna().unique()))
        disability_level = c6.selectbox("Disability Level (0=none)", sorted(patients_df["disability_level"].dropna().unique()))

        c7, c8, c9 = st.columns(3)
        has_hypertension = c7.selectbox("Hypertension?", ["No", "Yes"])
        has_diabetes = c8.selectbox("Diabetes?", ["No", "Yes"])
        has_alcoholism = c9.selectbox("Alcoholism?", ["No", "Yes"])

        st.subheader("Appointment")
        c10, c11, c12 = st.columns(3)
        clinic_id = c10.selectbox("Clinic", sorted(appointments_df["clinic_id"].dropna().unique()))
        doctor_id = c11.selectbox("Doctor", sorted(appointments_df["doctor_id"].dropna().unique()))
        specialty = c12.selectbox("Specialty", sorted(appointments_df["specialty"].dropna().unique()))

        c13, c14, c15 = st.columns(3)
        scheduled_date = c13.date_input("Scheduled Date", pd.Timestamp("2024-01-01"))
        appointment_date = c14.date_input("Appointment Date", pd.Timestamp("2024-01-15"))
        sms_reminder_sent = c15.selectbox("SMS Reminder Sent?", ["Yes", "No"])

        c16, c17, c18 = st.columns(3)
        weather_condition = c16.selectbox("Expected Weather", sorted(appointments_df["weather_condition"].dropna().unique()))
        payment_method = c17.selectbox("Payment Method", sorted(appointments_df["payment_method"].dropna().unique()))
        waitlist_flag = c18.selectbox("On Waitlist?", ["No", "Yes"])

        c19, c20, c21 = st.columns(3)
        temperature_c = c19.number_input("Temperature (°C)", -10.0, 50.0, 28.0)
        distance_to_clinic_km = c20.number_input("Distance to Clinic (km)", 0.0, 100.0, 5.0)
        appointment_cost = c21.number_input("Appointment Cost (Rs.)", 0.0, 50000.0, 2000.0)

        c22, c23 = st.columns(2)
        previous_appointments_count = c22.number_input("Previous Appointments", 0, 100, 2)
        previous_noshows_count = c23.number_input("Previous No-Shows", 0, 100, 0)

        submitted = st.form_submit_button("Predict No-Show Risk", use_container_width=True, type="primary")

    if submitted:
        raw = pd.DataFrame([{
            "age": age, "gender": gender, "neighborhood": neighborhood,
            "income_bracket": income_bracket, "insurance_type": insurance_type,
            "disability_level": disability_level,
            "has_hypertension": 1 if has_hypertension == "Yes" else 0,
            "has_diabetes": 1 if has_diabetes == "Yes" else 0,
            "has_alcoholism": 1 if has_alcoholism == "Yes" else 0,
            "clinic_id": clinic_id, "doctor_id": doctor_id, "specialty": specialty,
            "scheduled_date": scheduled_date, "appointment_date": appointment_date,
            "sms_reminder_sent": 1 if sms_reminder_sent == "Yes" else 0,
            "weather_condition": weather_condition, "payment_method": payment_method,
            "waitlist_flag": 1 if waitlist_flag == "Yes" else 0,
            "temperature_c": temperature_c, "distance_to_clinic_km": distance_to_clinic_km,
            "appointment_cost": appointment_cost,
            "previous_appointments_count": previous_appointments_count,
            "previous_noshows_count": previous_noshows_count,
        }])

        X = encode_batch(raw, FEATURE_NAMES)
        proba = calibrated_model.predict_proba(X)[0, 1]
        label, emoji = risk_bucket(proba)

        st.markdown("---")
        r1, r2 = st.columns([1, 2])
        with r1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                title={"text": f"No-Show Risk — {emoji} {label}"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "darkred" if proba > 0.4 else "darkorange" if proba > 0.2 else "seagreen"},
                       "steps": [
                           {"range": [0, 20], "color": "#d4edda"},
                           {"range": [20, 40], "color": "#fff3cd"},
                           {"range": [40, 60], "color": "#ffe5b4"},
                           {"range": [60, 100], "color": "#f8d7da"},
                       ]}))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            st.metric("Predicted No-Show Probability", f"{proba*100:.1f}%")
            if proba >= 0.40:
                st.warning(
                    "**High risk.** Consider: an extra reminder call, moving this slot "
                    "to a day with spare capacity, or offering it as an overbookable slot."
                )
            elif proba >= 0.20:
                st.info("**Moderate risk.** An SMS/call reminder is likely to help.")
            else:
                st.success("**Low risk.** Patient is likely to show up as scheduled.")

# --------------------------------------------------------------------------------------
# 4. BATCH PREDICTION
# --------------------------------------------------------------------------------------
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Prediction")
    st.markdown(
        "Upload a CSV of appointments (raw, unencoded — same shape as `appointments.csv` "
        "joined with patient info) and get back no-show probabilities for every row."
    )

    template_cols = [
        "age", "gender", "neighborhood", "income_bracket", "insurance_type",
        "has_hypertension", "has_diabetes", "has_alcoholism", "disability_level",
        "clinic_id", "doctor_id", "specialty", "scheduled_date", "appointment_date",
        "sms_reminder_sent", "weather_condition", "payment_method", "waitlist_flag",
        "temperature_c", "distance_to_clinic_km", "appointment_cost",
        "previous_appointments_count", "previous_noshows_count",
    ]
    sample = patients_df.merge(appointments_df, on="patient_id").sample(
        min(5, len(appointments_df)), random_state=1
    )
    template_csv = sample[template_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download template CSV (5 sample rows)", template_csv,
                        "batch_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload appointments CSV", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(raw)}** rows.")
        with st.spinner("Scoring appointments..."):
            X = encode_batch(raw, FEATURE_NAMES)
            proba = calibrated_model.predict_proba(X)[:, 1]
            out = raw.copy()
            out["predicted_noshow_proba"] = proba
            out["risk_level"] = [risk_bucket(p)[0] for p in proba]

        st.success("Done!")
        st.dataframe(out, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg. Predicted No-Show", f"{proba.mean()*100:.1f}%")
        c2.metric("High Risk (≥40%)", int((proba >= 0.4).sum()))
        c3.metric("Moderate (20-40%)", int(((proba >= 0.2) & (proba < 0.4)).sum()))
        c4.metric("Low Risk (<20%)", int((proba < 0.2).sum()))

        fig = px.histogram(out, x="predicted_noshow_proba", nbins=30,
                            title="Distribution of Predicted No-Show Probabilities")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("⬇️ Download predictions CSV",
                            out.to_csv(index=False).encode("utf-8"),
                            "predictions_output.csv", "text/csv", type="primary")

# --------------------------------------------------------------------------------------
# 5. OVERBOOKING SIMULATOR
# --------------------------------------------------------------------------------------
elif page == "🧮 Overbooking Simulator":
    st.title("🧮 Overbooking Policy Simulator")
    st.markdown(
        "Simulates clinic-days using the held-out test set (calibrated no-show "
        "probabilities + actual outcomes) and compares overbooking strategies."
    )

    test_df = load_test_predictions()
    default_revenue = float(test_df["appointment_cost"].median())

    c1, c2, c3 = st.columns(3)
    capacity = c1.slider("Daily Clinic Capacity", 5, 50, 20)
    revenue_per_patient = c2.number_input("Revenue per Patient Seen (Rs.)", 0.0, 50000.0, default_revenue)
    overflow_multiplier = c3.slider("Overflow Cost Multiplier", 1.0, 3.0, 1.5, 0.1)
    fixed_pct = st.slider("Fixed Overbooking %", 0, 50, 20) / 100

    cost_per_overflow = revenue_per_patient * overflow_multiplier

    if st.button("▶️ Run Simulation", type="primary"):
        shuffled = test_df.sample(frac=1, random_state=1).reset_index(drop=True)
        pool = test_df.sample(frac=1, random_state=99).reset_index(drop=True)

        with st.spinner("Simulating clinic-days..."):
            res_none = simulate_policy(shuffled, pool, capacity, revenue_per_patient,
                                        cost_per_overflow, policy="none")
            res_fixed = simulate_policy(shuffled, pool, capacity, revenue_per_patient,
                                         cost_per_overflow, policy="fixed", fixed_overbook_pct=fixed_pct)
            res_smart = simulate_policy(shuffled, pool, capacity, revenue_per_patient,
                                         cost_per_overflow, policy="smart")

        summary = pd.DataFrame([
            summarize(res_none, "No Overbooking"),
            summarize(res_fixed, f"Fixed Overbooking ({int(fixed_pct*100)}%)"),
            summarize(res_smart, "Smart Overbooking (Model-based)"),
        ])

        st.session_state["sim_summary"] = summary
        st.session_state["sim_days"] = len(res_none)

    if "sim_summary" in st.session_state:
        summary = st.session_state["sim_summary"]
        st.markdown(f"### Results ({st.session_state['sim_days']} simulated clinic-days)")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.bar(summary, x="Policy", y="Avg Utilization %", color="Policy",
                         title="Avg Utilization %")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(summary, x="Policy", y="Overcrowded Days %", color="Policy",
                         title="Overcrowded Days %", color_discrete_sequence=px.colors.sequential.Reds_r)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = px.bar(summary, x="Policy", y="Total Net Revenue", color="Policy",
                         title="Total Net Revenue", color_discrete_sequence=px.colors.sequential.Greens_r)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        best = summary.loc[summary["Total Net Revenue"].idxmax(), "Policy"]
        st.success(f"💡 **{best}** delivers the highest net revenue for these settings.")
    else:
        st.info("Set your parameters above and click **Run Simulation**.")

    with st.expander("📌 Reference: original notebook results (fixed parameters)"):
        orig_summary, _ = load_policy_summary()
        st.dataframe(orig_summary, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------------
# 6. MODEL PERFORMANCE
# --------------------------------------------------------------------------------------
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, roc_curve)

    test_df = load_test_predictions()
    y_true = test_df["actual_no_show"]
    y_proba = test_df["predicted_noshow_proba"]
    y_pred = (y_proba >= 0.5).astype(int)

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC (calibrated model)", f"{roc_auc_score(y_true, y_proba):.3f}")
    c2.metric("Avg Predicted Prob.", f"{y_proba.mean()*100:.1f}%")
    c3.metric("Actual No-Show Rate (test set)", f"{y_true.mean()*100:.1f}%")

    tab1, tab2, tab3 = st.tabs(["ROC Curve", "Confusion Matrix", "Feature Importance"])

    with tab1:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"Calibrated RF (AUC={roc_auc_score(y_true, y_proba):.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Guess", line=dict(dash="dash")))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         x=["Predicted: Show", "Predicted: No-Show"],
                         y=["Actual: Show", "Actual: No-Show"], title="Confusion Matrix (threshold 0.5)")
        st.plotly_chart(fig, use_container_width=True)
        st.text(classification_report(y_true, y_pred, target_names=["Show", "No-Show"]))

    with tab3:
        importances = pd.DataFrame({
            "feature": best_model.feature_names_in_,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False).head(15)
        fig = px.bar(importances.sort_values("importance"), x="importance", y="feature",
                     orientation="h", title="Top 15 Most Important Features (Random Forest)")
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • Random Forest (calibrated) • scikit-learn")
