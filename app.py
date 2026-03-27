import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from streamlit_option_menu import option_menu

# -------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title = "ChurnIQ",
    layout     = "centered"
)

# -------------------------------------------------------------------
# Load Artifacts
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)
    return model, encoders, scaler, feature_names

model, encoders, scaler, feature_names = load_artifacts()

# -------------------------------------------------------------------
# Prediction Function
# -------------------------------------------------------------------
def make_prediction(input_data: dict) -> tuple:
    input_df = pd.DataFrame([input_data])

    for column, encoder in encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    input_df = input_df[feature_names]

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    label = "Churn" if prediction == 1 else "No Churn"
    return label, round(float(probability), 4)

# -------------------------------------------------------------------
# Bulk CSV Prediction
# -------------------------------------------------------------------
def predict_from_row(row):
    input_data = {
        "gender"          : row["gender"],
        "SeniorCitizen"   : row["SeniorCitizen"],
        "Partner"         : row["Partner"],
        "Dependents"      : row["Dependents"],
        "tenure"          : row["tenure"],
        "PhoneService"    : row["PhoneService"],
        "MultipleLines"   : row["MultipleLines"],
        "InternetService" : row["InternetService"],
        "OnlineSecurity"  : row["OnlineSecurity"],
        "OnlineBackup"    : row["OnlineBackup"],
        "DeviceProtection": row["DeviceProtection"],
        "TechSupport"     : row["TechSupport"],
        "StreamingTV"     : row["StreamingTV"],
        "StreamingMovies" : row["StreamingMovies"],
        "Contract"        : row["Contract"],
        "PaperlessBilling": row["PaperlessBilling"],
        "PaymentMethod"   : row["PaymentMethod"],
        "MonthlyCharges"  : row["MonthlyCharges"],
        "TotalCharges"    : row["TotalCharges"],
    }
    try:
        label, prob = make_prediction(input_data)
        return label, round(prob * 100, 2)
    except Exception:
        return "Error", 0.0

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.title("ChurnIQ")

st.sidebar.markdown(
    """
**AI-powered Customer Churn Predictor**

This app uses a machine learning model trained on 7,043 telecom
customer records to predict whether a customer will churn or stay.

**Tech Stack:** Python, Scikit-learn, Random Forest, SMOTE, Streamlit
"""
)

st.sidebar.markdown("### Capabilities")
st.sidebar.markdown(
    """
- Single customer churn prediction
- Bulk CSV customer analysis
- Churn probability score
- Risk level assessment
- Downloadable results
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.markdown(
    """
- **Algorithm:** Random Forest  
- **Dataset:** IBM Telco Churn  
- **Records:** 7,043 customers  
- **Accuracy:** 78%  
- **ROC-AUC:** 0.83  
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
[GitHub](https://github.com/) |
[LinkedIn](https://linkedin.com/)
"""
)

# -------------------------------------------------------------------
# Title
# -------------------------------------------------------------------
st.title("ChurnIQ")

# -------------------------------------------------------------------
# Horizontal Navigation Menu
# -------------------------------------------------------------------
selected_tab = option_menu(
    menu_title    = None,
    options       = ["Single Prediction", "Bulk CSV Analysis", "About the Model"],
    icons         = ["person-check", "file-earmark-spreadsheet", "info-circle"],
    orientation   = "horizontal",
    default_index = 0
)

# -------------------------------------------------------------------
# TAB 1: Single Prediction
# -------------------------------------------------------------------
if selected_tab == "Single Prediction":
    st.subheader("Predict Churn for a Single Customer")
    st.write("Fill in the customer details below and click Predict to get the result.")

    st.divider()

    # Demographics
    st.markdown("#### Customer Demographics")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        senior_citizen = st.selectbox(
            "Senior Citizen",
            options     = [0, 1],
            format_func = lambda x: "Yes" if x == 1 else "No"
        )

    with col3:
        partner = st.selectbox("Has Partner", ["Yes", "No"])

    col4, col5 = st.columns(2)

    with col4:
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    with col5:
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    st.divider()

    # Services
    st.markdown("#### Phone and Internet Services")
    col6, col7, col8 = st.columns(3)

    with col6:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col7:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col8:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    col9, col10, col11 = st.columns(3)

    with col9:
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

    with col10:
        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

    with col11:
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

    col12, col13, col14 = st.columns(3)

    with col12:
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

    with col13:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

    with col14:
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

    st.divider()

    # Billing
    st.markdown("#### Account and Billing")
    col15, col16, col17 = st.columns(3)

    with col15:
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

    with col16:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col17:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col18, col19 = st.columns(2)

    with col18:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value = 0.0,
            max_value = 200.0,
            value     = 65.0,
            step      = 0.5
        )

    with col19:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value = 0.0,
            max_value = 10000.0,
            value     = float(monthly_charges * tenure) if tenure > 0 else 0.0,
            step      = 1.0
        )

    st.divider()

    if st.button("Predict", use_container_width=True, type="primary"):
        input_data = {
            "gender"          : gender,
            "SeniorCitizen"   : senior_citizen,
            "Partner"         : partner,
            "Dependents"      : dependents,
            "tenure"          : tenure,
            "PhoneService"    : phone_service,
            "MultipleLines"   : multiple_lines,
            "InternetService" : internet_service,
            "OnlineSecurity"  : online_security,
            "OnlineBackup"    : online_backup,
            "DeviceProtection": device_protection,
            "TechSupport"     : tech_support,
            "StreamingTV"     : streaming_tv,
            "StreamingMovies" : streaming_movies,
            "Contract"        : contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod"   : payment_method,
            "MonthlyCharges"  : monthly_charges,
            "TotalCharges"    : total_charges
        }

        with st.spinner("Running prediction..."):
            label, probability = make_prediction(input_data)
            percentage = round(probability * 100, 2)

        st.divider()
        st.markdown("#### Prediction Result")

        if label == "Churn":
            st.error(f"Prediction: This customer is likely to CHURN")
        else:
            st.success(f"Prediction: This customer is likely to STAY")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Churn Probability", f"{percentage}%")
        col_m2.metric("Prediction", label)
        col_m3.metric("Tenure", f"{tenure} months")

        st.progress(probability)

        st.divider()
        st.markdown("#### Risk Level")

        if percentage >= 70:
            st.error(
                "HIGH RISK — Immediate retention action required. "
                "Reach out with a personalized offer or discount."
            )
        elif percentage >= 40:
            st.warning(
                "MEDIUM RISK — Customer shows moderate churn signals. "
                "A proactive check-in is recommended."
            )
        else:
            st.success(
                "LOW RISK — Customer appears stable. "
                "Continue delivering consistent service quality."
            )

        st.divider()
        st.markdown("#### Risk Factors Detected")

        risk_flags = []

        if contract == "Month-to-month":
            risk_flags.append("Month-to-month contract — no long-term commitment")
        if internet_service == "Fiber optic":
            risk_flags.append("Fiber optic internet — highest churn segment")
        if payment_method == "Electronic check":
            risk_flags.append("Electronic check — associated with higher churn rate")
        if tenure < 12:
            risk_flags.append(f"Short tenure ({tenure} months) — new customers churn more")
        if monthly_charges > 70:
            risk_flags.append(f"High monthly charges (${monthly_charges:.2f})")
        if senior_citizen == 1:
            risk_flags.append("Senior citizen — slightly higher churn proportion")
        if online_security == "No" and internet_service != "No":
            risk_flags.append("No online security subscribed")
        if tech_support == "No" and internet_service != "No":
            risk_flags.append("No tech support subscribed")

        if risk_flags:
            for flag in risk_flags:
                st.markdown(f"- {flag}")
        else:
            st.markdown("No major risk factors detected.")

# -------------------------------------------------------------------
# TAB 2: Bulk CSV Analysis
# -------------------------------------------------------------------
if selected_tab == "Bulk CSV Analysis":
    st.subheader("Analyze Multiple Customers via CSV")
    st.write(
        "Upload a CSV file containing customer records. "
        "The file must have the same column names as the original dataset."
    )

    st.divider()

    st.markdown("**Required columns in your CSV:**")
    st.code(
        "gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, "
        "MultipleLines, InternetService, OnlineSecurity, OnlineBackup, "
        "DeviceProtection, TechSupport, StreamingTV, StreamingMovies, "
        "Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges"
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        required_columns = feature_names
        missing_cols     = [c for c in required_columns if c not in df.columns]

        if missing_cols:
            st.error(f"Missing columns in uploaded file: {missing_cols}")
        else:
            st.success(f"{len(df)} customer records loaded successfully.")
            st.dataframe(df.head())

            if st.button("Run Bulk Analysis", use_container_width=True, type="primary"):
                with st.spinner("Analyzing all customers..."):
                    results = df.apply(predict_from_row, axis=1)
                    df["Prediction"]      = results.apply(lambda x: x[0])
                    df["Churn Prob (%)"]  = results.apply(lambda x: x[1])

                st.success("Analysis complete.")
                st.divider()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Customers", len(df))
                col2.metric("Predicted Churn",    len(df[df["Prediction"] == "Churn"]))
                col3.metric("Predicted No Churn", len(df[df["Prediction"] == "No Churn"]))

                st.divider()
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label     = "Download Results as CSV",
                    data      = csv,
                    file_name = "churn_predictions.csv",
                    mime      = "text/csv",
                    use_container_width = True
                )

# -------------------------------------------------------------------
# TAB 3: About the Model
# -------------------------------------------------------------------
if selected_tab == "About the Model":
    st.subheader("Model Information")

    st.markdown(
        """
**Algorithm:** Random Forest Classifier  
**Dataset:** IBM Telco Customer Churn  
**Source:** Kaggle  
**Total Records:** 7,043 customers  
**Training Size:** 80% (5,634 records after SMOTE balancing)  
**Test Size:** 20% (1,409 records)  

**Model Performance:**

| Metric | Score |
|---|---|
| Accuracy | 78% |
| ROC-AUC | 0.83 |
| Precision (Churn) | 0.57 |
| Recall (Churn) | 0.66 |
| F1-Score (Churn) | 0.61 |

**Preprocessing Pipeline:**
- CustomerID removed (non-predictive)
- TotalCharges converted from string to float
- 15 categorical columns encoded with LabelEncoder
- 3 numerical columns scaled with StandardScaler
- Class imbalance handled with SMOTE (Synthetic Minority Oversampling)

**Hyperparameter Tuning:**
- Method: GridSearchCV with 5-fold cross-validation
- Best Parameters: n_estimators=200, max_depth=None

**Top Churn Indicators (from Feature Importance):**
- tenure
- MonthlyCharges
- TotalCharges
- Contract type
- Internet service type
"""
    )