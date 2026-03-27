# Customer Churn Prediction

A complete end-to-end machine learning project that predicts whether a telecom customer is likely to churn, based on their demographics, account details, and subscribed services. The project covers data exploration, preprocessing, model training, evaluation, and deployment through a Streamlit web application.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Notebook Sections](#notebook-sections)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Performance](#model-performance)
- [Installation and Setup](#installation-and-setup)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Deployment Files](#deployment-files)
- [Key Findings from EDA](#key-findings-from-eda)
- [Technologies Used](#technologies-used)

---

## Project Overview

Customer churn refers to the loss of clients or customers. For telecom companies, retaining existing customers is significantly more cost-effective than acquiring new ones. This project builds a binary classification model to identify customers who are at risk of churning so that the business can take proactive retention measures.

The workflow is divided into two phases:

- **Phase 1 — Analysis and Training:** Conducted in Google Colab using a Jupyter notebook. Covers all steps from raw data loading to saving the final trained model.
- **Phase 2 — Deployment:** A Streamlit application built on a local machine that loads the saved model and allows real-time predictions through an interactive interface.

---

## Dataset

**Source:** IBM Telco Customer Churn Dataset  
**Platform:** Kaggle  
**Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

| Property | Value |
|---|---|
| Total Records | 7,043 customers |
| Total Features | 21 columns |
| Target Variable | Churn (Yes / No) |
| Class Distribution | No Churn: 73.5%, Churn: 26.5% |

### Feature Categories

**Demographics**
- gender, SeniorCitizen, Partner, Dependents

**Account Information**
- tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

**Services Subscribed**
- PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

---

## Project Structure

```
churn_app/
|
|-- Customer_Churn_Prediction.ipynb    # Main analysis and training notebook
|-- app.py                             # Streamlit web application
|-- requirements.txt                   # Python dependencies
|
|-- best_model.pkl                     # Trained Random Forest model
|-- encoder.pkl                        # LabelEncoders for categorical columns
|-- scaler.pkl                         # StandardScaler for numerical columns
|-- feature_names.json                 # Feature column order used during training
|
|-- README.md                          # Project documentation
```

---

## Notebook Sections

The Jupyter notebook is structured into eight sections, each clearly separated with markdown explanations.

| Section | Title | Description |
|---|---|---|
| 1 | Importing Libraries | All required libraries imported and versions verified |
| 2 | Data Loading and Initial Exploration | Dataset loaded from Kaggle, shape, types, and hidden nulls examined |
| 3 | Exploratory Data Analysis | 10 individual visualizations covering numerical and categorical features |
| 4 | Data Preprocessing | Column removal, type fixing, encoding, scaling, and artifact saving |
| 5 | Model Training | Train-test split, SMOTE balancing, GridSearchCV tuning for two models |
| 6 | Model Evaluation | Accuracy, ROC-AUC, confusion matrix, ROC curve, and feature importance |
| 7 | Prediction Function | Reusable function tested on three customer profiles before deployment |
| 8 | Streamlit Deployment | Complete app code, setup instructions, and requirements |

---

## Machine Learning Pipeline

### Step 1 — Preprocessing

- Removed the `customerID` column as it carries no predictive value
- Fixed `TotalCharges` column which was stored as string with blank spaces for new customers
- Encoded `Churn` target variable: Yes to 1, No to 0
- Applied `LabelEncoder` individually to each of the 15 categorical columns and saved all encoders
- Applied `StandardScaler` to the three numerical columns: tenure, MonthlyCharges, TotalCharges

### Step 2 — Handling Class Imbalance

The dataset has a 73/27 split between retained and churned customers. Training directly on this imbalanced data would bias the model toward the majority class.

SMOTE (Synthetic Minority Oversampling Technique) was applied exclusively on the training set to balance both classes to equal counts. The test set was left untouched to reflect real-world conditions.

| | Before SMOTE | After SMOTE |
|---|---|---|
| No Churn (0) | 4,138 | 4,138 |
| Churn (1) | 1,496 | 4,138 |

### Step 3 — Model Training

Two models were trained and compared using GridSearchCV with 5-fold cross-validation.

**Random Forest**

| Hyperparameter | Values Searched | Best Value |
|---|---|---|
| n_estimators | 50, 100, 200 | 200 |
| max_depth | 5, 10, None | None |

**XGBoost**

| Hyperparameter | Values Searched | Best Value |
|---|---|---|
| learning_rate | 0.01, 0.1, 0.2 | 0.1 |
| max_depth | 3, 5, 7 | 7 |

---

## Model Performance

Both models were evaluated on the held-out test set of 1,409 records.

| Metric | Random Forest | XGBoost |
|---|---|---|
| Accuracy | 78% | 77% |
| ROC-AUC | 0.83+ | 0.82+ |
| Precision (Churn) | 0.57 | 0.56 |
| Recall (Churn) | 0.66 | 0.65 |
| F1-Score (Churn) | 0.61 | 0.60 |

**Selected Model for Deployment:** Random Forest

**Reason:** Achieves the best balance between Precision and Recall on the minority class (Churn = 1) with the highest ROC-AUC score. ROC-AUC was prioritized over raw accuracy because the dataset is imbalanced.

### Important Note on Evaluation Metrics

For churn prediction, Recall on the positive class (Churn = 1) is the most business-critical metric. A false negative — predicting that a customer will stay when they actually leave — is more costly than a false positive. The model was selected and evaluated with this priority in mind.

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- A Kaggle account (for downloading the dataset in the notebook)

### Step 1 — Clone or Download the Project

Download all project files and place them in a single folder on your local machine named `churn_app`.

### Step 2 — Install Dependencies

Open a terminal inside the `churn_app` folder and run:

```bash
pip install -r requirements.txt
```

### Contents of requirements.txt

```
streamlit
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
joblib
```

### Step 3 — Verify Deployment Files

Confirm that the following four files are present in the `churn_app` folder before launching the app:

- `best_model.pkl`
- `encoder.pkl`
- `scaler.pkl`
- `feature_names.json`

These files are generated and downloaded at the end of the Jupyter notebook (Section 7). If they are missing, rerun the notebook and download them again.

---

## Running the Streamlit App

Once all dependencies are installed and all four deployment files are present, launch the application with:

```bash
streamlit run app.py
```

The app will open automatically in your default browser at:

```
http://localhost:8501
```

### Using the Application

The interface is divided into three input sections:

1. **Customer Demographics** — gender, senior citizen status, partner, dependents, tenure
2. **Phone and Internet Services** — phone service, multiple lines, internet type, online security, backup, device protection, tech support, streaming services
3. **Account and Billing Information** — contract type, paperless billing, payment method, monthly charges, total charges

After filling in all fields, click the **Predict Churn** button. The app will display:

- Prediction label: Churn or No Churn
- Churn probability as a percentage
- Risk level assessment: Low, Medium, or High
- Key risk factors detected based on the customer profile
- A full summary table of all entered values

---

## Deployment Files

The following files are produced by the notebook and required by the Streamlit application. They must all be in the same directory as `app.py`.

| File | Type | Purpose |
|---|---|---|
| best_model.pkl | Pickle | Trained Random Forest classifier with best hyperparameters |
| encoder.pkl | Pickle | Dictionary of 15 LabelEncoder objects, one per categorical column |
| scaler.pkl | Pickle | StandardScaler fitted on training data numerical columns |
| feature_names.json | JSON | Ordered list of 19 feature names matching the training column sequence |

### Why All Four Files Are Needed

When a user submits customer data through the app, the input must go through the exact same transformation pipeline that was applied during training. The encoders convert text categories to numbers using the same mapping. The scaler normalizes numerical values using the same mean and standard deviation. The feature names ensure the columns are passed to the model in the correct order. Any mismatch in these steps would produce incorrect predictions.

---

## Key Findings from EDA

The exploratory data analysis revealed several strong patterns associated with customer churn:

**Contract Type** is the most influential factor. Month-to-month customers churn at a significantly higher rate than customers on one-year or two-year contracts. Long-term contracts create commitment that reduces churn.

**Tenure** has a strong inverse relationship with churn. Customers who have been with the company for less than 12 months are far more likely to leave. Churn risk decreases consistently as tenure increases.

**Internet Service Type** shows that fiber optic customers churn more than DSL or no-internet customers. This may indicate dissatisfaction with pricing or service quality in the fiber optic segment.

**Payment Method** reveals that customers paying by electronic check have a notably higher churn rate compared to those using automatic payment methods such as bank transfer or credit card.

**Monthly Charges** indicates that churned customers tend to pay higher monthly charges. Higher-paying customers may be more price-sensitive and more likely to evaluate competitor offerings.

**Senior Citizens**, while a minority in the dataset, show a disproportionately higher churn proportion relative to non-senior customers.

These findings informed the feature importance results from the Random Forest model, where tenure, MonthlyCharges, TotalCharges, and Contract type ranked as the top predictive features.

---

## Technologies Used

| Category | Library / Tool |
|---|---|
| Language | Python 3.8 |
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Gradient Boosting | xgboost |
| Class Balancing | imbalanced-learn (SMOTE) |
| Model Serialization | pickle |
| Web Application | streamlit |
| Notebook Environment | Google Colab |
| Dataset Source | Kaggle |

---

## Author Notes

This project was built as a complete end-to-end demonstration of a production-style machine learning workflow. The notebook is written to be readable and self-explanatory — each section begins with a markdown description of what is being done and why, followed by well-commented code cells.

The Streamlit application is designed to mirror the exact preprocessing pipeline from the notebook so that predictions are consistent and reliable. All saved artifacts are loaded once at startup using caching to ensure the app remains responsive.
