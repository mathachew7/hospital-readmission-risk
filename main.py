import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

# === Load Datasets ===
df = pd.read_csv('./data/diabetic_data.csv')
mapping_df = pd.read_csv('./data/IDS_mapping.csv', header=None, skip_blank_lines=False)

# === Clean Data ===
df.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True)
df = df[df['race'] != '?']
df = df[df['gender'] != 'Unknown/Invalid']
df = df[df['diag_1'] != '?']
df = df[df['readmitted'] != '>30']
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# === Map Admission/Discharge/Source IDs ===
map_dicts = {}
current_section = None
for _, row in mapping_df.iterrows():
    key = str(row.iloc[0]).strip()
    value = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else None
    if key in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        current_section = key
        map_dicts[current_section] = {}
    elif current_section and key.isdigit():
        map_dicts[current_section][int(key)] = value

df['admission_type'] = df['admission_type_id'].map(map_dicts['admission_type_id'])
df['discharge_dest'] = df['discharge_disposition_id'].map(map_dicts['discharge_disposition_id'])
df['admission_source'] = df['admission_source_id'].map(map_dicts['admission_source_id'])
df.drop(['admission_type_id', 'discharge_disposition_id', 'admission_source_id'], axis=1, inplace=True)

# === Encode Categorical Variables ===
for col in ['admission_type', 'discharge_dest', 'admission_source']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# === Feature Engineering ===
df['elderly'] = df['age'].apply(lambda x: 1 if x >= 8 else 0)
df['long_stay'] = (df['time_in_hospital'] > 7).astype(int)
df['high_lab_use'] = (df['num_lab_procedures'] > 60).astype(int)
df['high_med_use'] = (df['num_medications'] > 12).astype(int)
df['multi_diag'] = (df['number_diagnoses'] > 5).astype(int)
df['multiple_visits'] = ((df['number_inpatient'] + df['number_emergency'] + df['number_outpatient']) > 2).astype(int)
df['no_med_change'] = (df['change'] == 0).astype(int)
df['has_insulin'] = df['insulin'].apply(lambda x: 1 if x in [1, 2] else 0)

# Drop known low-signal features
low_signal = ['examide', 'citoglipton', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
df.drop(columns=[col for col in low_signal if col in df.columns], inplace=True)

# === Prepare Data ===
X = df.drop('readmitted', axis=1)
y = df['readmitted']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === L1 Logistic Regression for Feature Selection ===
lasso = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=2000)
lasso.fit(X_train, y_train)

# Select only non-zero coef features
selector = SelectFromModel(lasso, prefit=True)
X_train_l1 = selector.transform(X_train)
X_test_l1 = selector.transform(X_test)
selected_features = np.array(X.columns)[selector.get_support()]

# === Final Logistic Regression ===
final_model = LogisticRegression(class_weight='balanced', max_iter=2000)
final_model.fit(X_train_l1, y_train)
y_pred = final_model.predict(X_test_l1)
y_prob = final_model.predict_proba(X_test_l1)[:, 1]

# === Evaluation ===
report = classification_report(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
p, r, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(r, p)

# === Results ===
print("📊 CLASSIFICATION REPORT:")
print(report)
print(f"🎯 ROC AUC: {roc:.3f}")
print(f"🤝 PR-AUC: {pr_auc:.3f}")
print(f"🧠 Features Used ({len(selected_features)}): {selected_features.tolist()}")

# === Plot PR Curve ===
plt.figure(figsize=(8,6))
plt.plot(r, p, label=f'PR-AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve – Final Logistic Regression')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
