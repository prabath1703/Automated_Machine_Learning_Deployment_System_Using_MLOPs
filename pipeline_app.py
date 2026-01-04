from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import glob
import os

# ------------------ Find latest best model ------------------
all_models = glob.glob("artifacts/best_model_*")
if not all_models:
    raise ValueError("No saved best model found in artifacts/")

latest_model_dir = max(all_models, key=os.path.getctime)
print(f"Loading latest best model from: {latest_model_dir}")

model = mlflow.pyfunc.load_model(latest_model_dir)

# ------------------ FastAPI ------------------
app = FastAPI(title="Student Pass Prediction API")

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudentInput(BaseModel):
    study_hours: float
    attendance: float
    internal_marks: float

@app.get("/")
def health():
    return {"status": "Model API is running", "model_path": latest_model_dir}

@app.post("/predict")
def predict(data: StudentInput):
    input_df = pd.DataFrame([{
        "study_hours": data.study_hours,
        "attendance": data.attendance,
        "internal_marks": data.internal_marks
    }])
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}
