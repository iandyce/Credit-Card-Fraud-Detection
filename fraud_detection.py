import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Show dataset shape
print("Shape of dataset:", df.shape)

# Fraud vs Normal transactions
class_counts = df['Class'].value_counts()
print("\nFraud vs Normal transactions:")
print(class_counts)

# --- Visualization 1: Fraud vs Normal Count ---
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df, palette="coolwarm")
plt.title("Fraud vs Normal Transactions")
plt.xticks([0,1], ["Normal (0)", "Fraud (1)"])
plt.show()

# --- Visualization 2: Transaction Amount Distribution ---
plt.figure(figsize=(8,5))
sns.histplot(df[df['Class']==0]['Amount'], bins=100, color='blue', label="Normal", alpha=0.6)
sns.histplot(df[df['Class']==1]['Amount'], bins=100, color='red', label="Fraud", alpha=0.6)
plt.legend()
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount ($)")
plt.show()

# --- Visualization 3: Transactions over Time ---
plt.figure(figsize=(8,5))
sns.histplot(df[df['Class']==0]['Time'], bins=100, color='blue', label="Normal", alpha=0.6)
sns.histplot(df[df['Class']==1]['Time'], bins=100, color='red', label="Fraud", alpha=0.6)
plt.legend()
plt.title("Transactions Over Time")
plt.xlabel("Time (seconds)")
plt.show()

# --- Visualization 4: Correlation Heatmap ---
plt.figure(figsize=(12,8))
corr = df.corr()

# Focus on correlations with the target 'Class'
corr_with_class = corr['Class'].sort_values(ascending=False)
print("\nCorrelation of features with Fraud (Class):\n")
print(corr_with_class)

sns.heatmap(corr, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --- Undersampling ---
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0].sample(n=len(fraud), random_state=42)  # take equal number of normal cases

df_balanced = pd.concat([fraud, normal])
print("\nAfter undersampling:")
print(df_balanced['Class'].value_counts())

# Features & target
X = df_balanced.drop('Class', axis=1)
y = df_balanced['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

from sklearn.ensemble import RandomForestClassifier

# --- Random Forest Classifier ---
rf_model = RandomForestClassifier(
    n_estimators=100,        # number of trees
    class_weight="balanced", # handle imbalance
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest Results ===")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, digits=4))

from imblearn.over_sampling import SMOTE

# --- SMOTE Oversampling ---
X = df.drop('Class', axis=1)
y = df['Class']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nAfter SMOTE Oversampling:")
print(y_resampled.value_counts())  # should now be equal classes

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)

# Logistic Regression with SMOTE
smote_log_model = LogisticRegression(max_iter=1000)
smote_log_model.fit(X_train, y_train)
y_pred_smote_log = smote_log_model.predict(X_test)

print("\n=== Logistic Regression with SMOTE Results ===")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_smote_log))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_smote_log, digits=4))

# Random Forest with SMOTE
smote_rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)
smote_rf_model.fit(X_train, y_train)
y_pred_smote_rf = smote_rf_model.predict(X_test)

print("\n=== Random Forest with SMOTE Results ===")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_smote_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_smote_rf, digits=4))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =========================
# 1. Scale Features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # scale original features before splitting

# =========================
# 2. Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Original Train distribution:\n", y_train.value_counts())
print("Original Test distribution:\n", y_test.value_counts())

# =========================
# 3. Apply SMOTE on Training Set ONLY
# =========================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE on Train set:\n", y_train_res.value_counts())

# =========================
# 4. Logistic Regression with SMOTE
# =========================
log_reg = LogisticRegression(max_iter=5000, solver='lbfgs')
log_reg.fit(X_train_res, y_train_res)

y_pred_lr = log_reg.predict(X_test)
y_pred_proba_lr = log_reg.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression with SMOTE on Test Set ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# =========================
# 5. Random Forest with SMOTE
# =========================
rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=200)
rf.fit(X_train_res, y_train_res)

y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest with SMOTE on Test Set ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# =========================
# 6. ROC Curves
# =========================
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)

auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# =========================
# 7. Accuracy & AUC summary
# =========================
print("\n=== Summary ===")
print(f"Logistic Regression: Accuracy = {accuracy_score(y_test, y_pred_lr):.4f}, AUC = {auc_lr:.4f}")
print(f"Random Forest:       Accuracy = {accuracy_score(y_test, y_pred_rf):.4f}, AUC = {auc_rf:.4f}")

# =========================
# 8. Random Forest Feature Importance
# =========================
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Random Forest - Top 10 Feature Importances")
plt.bar(range(10), importances[indices[:10]], align="center")
plt.xticks(range(10), [df.drop("Class", axis=1).columns[i] for i in indices[:10]], rotation=45)
plt.tight_layout()
plt.show()

# =========================
# 9. Save Models
# =========================
joblib.dump(log_reg, "logistic_regression_model.pkl")
joblib.dump(rf, "random_forest_model.pkl")
print("\nModels saved as 'logistic_regression_model.pkl' and 'random_forest_model.pkl'")
