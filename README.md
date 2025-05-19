# Predicting 30-Day Hospital Readmissions Using Interpretable Machine Learning

## 🔍 Project Overview
This project uses real-world electronic health records from over 130 U.S. hospitals to build a predictive model for identifying patients at risk of being readmitted within 30 days of discharge. The primary focus is on balancing **accuracy**, **recall**, and **clinical interpretability** — essential in healthcare applications.

## 📊 Business Goal
- **Objective**: Flag high-risk patients for early intervention to reduce costly readmissions.
- **Value**: Enables smarter discharge planning, reduces financial penalties, and improves patient care quality.

## 📁 Dataset
- **Source**: UCI ML Repository – Diabetes 130-US hospitals data
- **Records**: 101,766 unique admissions
- **Features**: Demographics, lab procedures, diagnoses, medications, and discharge information

## 🧠 Modeling Approach
- **Model Type**: Logistic Regression (interpretable, healthcare-trusted)
- **Feature Engineering**:
  - Flags for high lab use, polypharmacy, long stay
  - Behavioral patterns like frequent visits and medication changes
  - Mapped `admission_type_id`, `discharge_disposition_id`, and `admission_source_id` to readable formats
- **Feature Selection**: L1-penalized logistic regression (LASSO) to drop irrelevant features

## ✅ Final Model Metrics
| Metric | Value |
|--------|-------|
| Recall (Class 1 – Readmitted) | 0.55 |
| Precision (Class 1) | 0.29 |
| F1 Score (Class 1) | 0.38 |
| Accuracy | 0.70 |
| ROC AUC | 0.699 |
| PR AUC | 0.361 |
| Features Used | 46 (LASSO-selected from 60+) |

## 📌 Key Features Used
- `time_in_hospital`, `num_lab_procedures`, `number_inpatient`
- `diag_1`, `diag_2`, `diag_3`, `age`, `insulin`, `diabetesMed`
- Engineered features: `elderly`, `multi_diag`, `long_stay`, `multiple_visits`, `has_insulin`, `no_med_change`

## 📉 Confusion Matrix Summary
|               | Predicted: No | Predicted: Yes |
|---------------|----------------|-----------------|
| Actual: No    | 7,792          | 2,913           |
| Actual: Yes   | 990            | 1,199           |

## 📈 Visualizations
- Precision-Recall curve showing reliable signal in imbalanced data
- Confusion matrix to visualize recall-heavy performance

## ⚙️ How to Run

### 1. Clone this repository
```bash
git clone https://github.com/your-username/hospital-readmission-prediction.git
cd hospital-readmission-prediction

```

### 2. Install dependencies
```bash
pip install -r requirements.txt

```

### 3. Prepare your data 
```bash
- Place diabetic_data.csv and IDS_mapping.csv in the /data folder
```

### 4. Run the pipeline
```bash
python main.py
```

---

## 📬 Contact

Built by @mathachew7 for academic, clinical, and public interest ML applications. Open to collaboration and feedback.

