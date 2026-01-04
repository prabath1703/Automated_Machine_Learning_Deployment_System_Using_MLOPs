import os
import time
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ------------------ Paths & Config ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src folder
DATASET_PATH = os.path.join(BASE_DIR, "../data/student.csv")
REGISTERED_MODEL_NAME = "student_pass_model"
EXPORT_ARTIFACTS = True  # save local copy of best model
MLRUNS_PATH = os.path.join(BASE_DIR, "../mlruns")  # ensure repo-root relative

# ------------------ Load Dataset ------------------
data = pd.read_csv(DATASET_PATH)

# Required columns
expected_cols = ["study_hours", "attendance", "internal_marks", "pass"]
missing_cols = [c for c in expected_cols if c not in data.columns]
if missing_cols:
    raise ValueError(f"Dataset missing columns: {missing_cols}")

data = data[expected_cols]

# Encode target
data["pass"] = LabelEncoder().fit_transform(data["pass"])

X = data.drop("pass", axis=1)
y = data["pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ MLflow Setup ------------------
mlflow.set_tracking_uri(f"file:{MLRUNS_PATH}")
mlflow.set_experiment("Best_Student_Model")

results = {}

# ------------------ Logistic Regression ------------------
with mlflow.start_run(run_name="Logistic_Regression"):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(lr, name="model")

    results["Logistic Regression"] = (lr, acc)

# ------------------ Random Forest ------------------
with mlflow.start_run(run_name="Random_Forest"):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(rf, name="model")

    results["Random Forest"] = (rf, acc)

# ------------------ Select Best Model ------------------
best_model_name = max(results, key=lambda x: results[x][1])
best_model, best_accuracy = results[best_model_name]

print("Training completed ✅")
print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy}")

# ------------------ Register Best Model ------------------
with mlflow.start_run(run_name="Register_Best_Model"):
    mlflow.log_param("selected_model", best_model_name)
    mlflow.log_metric("best_accuracy", best_accuracy)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="model",
        registered_model_name=REGISTERED_MODEL_NAME
    )

# ------------------ Get new version ------------------
client = MlflowClient()
latest_versions = client.get_latest_versions(REGISTERED_MODEL_NAME)
new_version = latest_versions[-1].version
print(f"Registered version '{new_version}' of model '{REGISTERED_MODEL_NAME}'")

# ------------------ Auto-Promote to Production ------------------
def get_production_accuracy(model_name):
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        return None, None
    prod_version = versions[0]
    run_id = prod_version.run_id
    run = client.get_run(run_id)
    return prod_version.version, run.data.metrics.get("accuracy")

prod_version, prod_accuracy = get_production_accuracy(REGISTERED_MODEL_NAME)

print("Current Production Accuracy:", prod_accuracy)
print("New Model Accuracy:", best_accuracy)

if prod_accuracy is None or best_accuracy > prod_accuracy:
    print("Promoting new model to Production ✅")
    if prod_version:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=prod_version,
            stage="Archived"
        )
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=new_version,
        stage="Production"
    )
else:
    print("New model is not better. Keeping current Production model.")

# ------------------ Optional: Save local copy ------------------
if EXPORT_ARTIFACTS:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPORT_DIR = os.path.join(BASE_DIR, f"../artifacts/best_model_{timestamp}")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    mlflow.sklearn.save_model(best_model, path=EXPORT_DIR)
    print(f"Local export saved at: {EXPORT_DIR}")
