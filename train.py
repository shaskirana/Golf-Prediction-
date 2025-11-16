#ini merupakan coding versi final setelah melalui berbagai proses percobaan pada folder notebook
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# ==============================
# 1. Load dataset
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "golf_dataset_long_format_with_text.csv")

print(f"[INFO] Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ==============================
# 2. Pilih fitur & target
# ==============================

feature_cols = [
    "Weekday",
    "Holiday",
    "Month",
    "Season",
    "Temperature",
    "Humidity",
    "Windy",
    "Outlook",
    "Crowdedness",
    "EmailCampaign",
    "ID",
]

target_col = "Play"

X = df[feature_cols].copy()
y = df[target_col].astype(int)

# ==============================
# 3. Oversampling manual (balance)
# ==============================

df_model = X.copy()
df_model["Play"] = y

df_major = df_model[df_model["Play"] == 0]
df_minor = df_model[df_model["Play"] == 1]

print("\n[INFO] Before balance:")
print(df_model["Play"].value_counts())

df_minor_upsampled = df_minor.sample(
    n=len(df_major),
    replace=True,
    random_state=42
)

df_balanced = pd.concat([df_major, df_minor_upsampled]).sample(
    frac=1,
    random_state=42
)

print("\n[INFO] After balance:")
print(df_balanced["Play"].value_counts())

X_bal = df_balanced[feature_cols]
y_bal = df_balanced["Play"].astype(int)

# ==============================
# 4. Train-test split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_bal,
    y_bal,
    test_size=0.2,
    stratify=y_bal,
    random_state=42,
)

print("\n[INFO] Train shape:", X_train.shape)
print("[INFO] Test shape :", X_test.shape)

# ==============================
# 5. Preprocessing
# ==============================

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

print("\n[INFO] Numerical columns   :", num_cols)
print("[INFO] Categorical columns :", cat_cols)

numeric_tf = SimpleImputer(strategy="mean")

categorical_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols),
    ]
)

# ==============================
# 6. Random Forest (anti-overfit)
#    -> param samain dengan yang di notebook
# ==============================

rf_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=40,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )),
])

print("\n[INFO] Training Random Forest (final, regularized)...")
rf_clf.fit(X_train, y_train)

# ==============================
# 7. Evaluasi
# ==============================

y_pred = rf_clf.predict(X_test)
y_prob = rf_clf.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_prob)

print("\n=== Final Model Evaluation ===")
print("Test Accuracy :", test_acc)
print("Test ROC-AUC  :", test_auc)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# ==============================
# 8. Simpan model
# ==============================

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "golf_model.pkl")

joblib.dump(rf_clf, MODEL_PATH)
print(f"\n[SAVED] Model saved to: {MODEL_PATH}")
