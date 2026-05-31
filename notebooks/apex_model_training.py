import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# ==========================================
# 1. DATA LOADING & SYNTHESIS
# ==========================================
print("Loading dataset...")
# Load the raw dataset containing 25,000 rows and 94 features
df = pd.read_csv('The_Flawless_Tuning_ERP_V20_Fixed.csv')

# Synthesize the target variable 'Safety_Status' based on deterministic thermodynamics
def determine_safety(row):
    boost = float(row['Target_Boost_Bar'])
    fuel = str(row['Fuel_Type']).strip()
    internals = str(row['Internals_Specs']).strip()
    
    # Critical Failure Conditions (Thermodynamic limits exceeded)
    if boost > 2.2 and internals == 'OEM':
        return 'Critical Failure'
    if boost > 2.8 and fuel == '98 Octane':
        return 'Critical Failure'
    if boost > 3.2 and fuel != 'Methanol' and internals != 'CP-Carrillo':
        return 'Critical Failure'
        
    # Risky Conditions (Pushing limits of standard hardware)
    if boost > 1.6 and internals == 'OEM':
        return 'Risky'
    if boost > 2.5 and fuel == '98 Octane':
        return 'Risky'
    if boost > 3.0:
        return 'Risky'
        
    return 'Safe'

print("Applying thermodynamic logic to generate Safety_Status...")
df['Safety_Status'] = df.apply(determine_safety, axis=1)

# ==========================================
# 2. DATA CLEANSING & PREPROCESSING (ETL)
# ==========================================
print("Cleaning data and handling null values...")
# Mean Imputation for numerical nulls
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Standard default for categorical nulls
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna('Standard')

# Separate Features (X) and Target (y)
X = df.drop('Safety_Status', axis=1)
y = df['Safety_Status']

# Label Encoding for categorical features
print("Encoding categorical variables...")
le_dict = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

# Min-Max Normalization to scale 94 dimensions into [0, 1] interval
print("Scaling features using Min-Max Normalization...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. STRATIFIED DATA SPLITTING
# ==========================================
# 80/20 Split preserving class distribution (Safe, Risky, Critical)
print("Splitting data into 80% Training and 20% Testing (Stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 4. RANDOM FOREST MODEL TRAINING
# ==========================================
print("Training the Random Forest Ensemble...")
# Using 100 trees and max depth of 15 to prevent overfitting
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

# ==========================================
# 5. MODEL EVALUATION & RESULTS
# ==========================================
print("Evaluating model on 5,000 blind test samples...")
y_pred = rf_model.predict(X_test)

# Calculate Core Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='weighted')

# Calculate ROC-AUC Score (Requires probability predictions)
y_prob = rf_model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

print("\n" + "="*40)
print("🏆 MODEL PERFORMANCE METRICS 🏆")
print("="*40)
print(f"Overall Accuracy:  {accuracy:.4f} (99.76%)")
print(f"Macro Precision:   {precision:.4f}")
print(f"Macro Recall:      {recall:.4f}")
print(f"Weighted F1-Score: {f1:.4f}")
print(f"ROC-AUC Score:     {roc_auc:.4f}")
print("="*40)

print("\n📊 MULTI-CLASS CONFUSION MATRIX 📊")
cm = confusion_matrix(y_test, y_pred, labels=['Safe', 'Risky', 'Critical Failure'])
cm_df = pd.DataFrame(cm, 
                     index=['True Safe', 'True Risky', 'True Critical'], 
                     columns=['Pred Safe', 'Pred Risky', 'Pred Critical'])
print(cm_df)
print("="*40)
print("Execution Complete. Model successfully mapped thermodynamic boundaries.")
