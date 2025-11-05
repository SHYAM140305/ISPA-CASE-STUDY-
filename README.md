# 📊 Customer Churn Prediction — Case Study (21AIC401T)

End-to-end churn prediction using the IBM Telco Customer Churn dataset.  
Includes data preprocessing, exploratory analysis, model building with Logistic Regression and CHAID (Decision-Tree equivalent), model comparison, evaluation, and deployment simulation.

---

### 📘 Overview
This project predicts telecom customer churn using **Python (scikit-learn)**.  
The goal is to identify customers likely to leave and help design retention strategies.

**Key steps**
- Data cleaning & preprocessing  
- Exploratory Data Analysis (EDA)  
- CHAID-like Decision Tree model  
- Logistic Regression baseline  
- Model comparison (Accuracy, ROC-AUC, Lift, Gains)  
- Model deployment & inference demo  

---

### 🗂 Dataset
**Source:** [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)  
7,043 records • 21 features • Target → `Churn (Yes/No)`

---

### ⚙️ Tech Stack
Python • pandas • numpy • scikit-learn • matplotlib • joblib  

---

### 📈 Results
| Model | Accuracy | ROC-AUC |
|--------|-----------|----------|
| Decision Tree (CHAID-like) | 0.77 | 0.78 |
| Logistic Regression | 0.80 | 0.84 |

**ROC Curve and Lift Chart** show that Logistic Regression performs better overall, while the Decision Tree offers strong interpretability.

---

### 🧩 Quick Inference Example
```python
import joblib, pandas as pd
model = joblib.load("artifacts/logreg_churn_pipeline.joblib")

sample = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.35,
    "TotalCharges": 350.5
}

df = pd.DataFrame([sample])
proba = model.predict_proba(df)[0][1]
pred = model.predict(df)[0]
print(f"Churn Probability: {proba:.2f}")
print("Prediction:", "Churn" if pred == 1 else "No Churn")
