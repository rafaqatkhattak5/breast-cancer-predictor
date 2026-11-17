import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.datasets import load_breast_cancer
from tensorflow import keras

BASE = os.path.dirname(os.path.abspath(__file__))

model = keras.models.load_model(os.path.join(BASE, "breast_cancer_model.keras"))
scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
HTML_PAGE = open(os.path.join(BASE, "index.html"), encoding="utf-8").read()
raw = load_breast_cancer()

app = FastAPI(title="Breast Cancer Predictor")

class Patient(BaseModel):
    features: list[float]

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

@app.get("/features")
def features():
    return {"features": list(raw.feature_names)}

@app.get("/sample")
def sample(kind: str = "malignant"):
    label = 0 if kind == "malignant" else 1
    rows = raw.data[raw.target == label]
    row = rows[np.random.randint(len(rows))]
    return {"features": row.tolist()}

@app.post("/predict")
def predict(p: Patient):
    if len(p.features) != 30:
        raise HTTPException(status_code=400, detail="Exactly 30 features required")
    x = scaler.transform(np.array(p.features).reshape(1, -1))
    prob_benign = float(model.predict(x, verbose=0)[0][0])
    return {
        "label": "benign" if prob_benign > 0.5 else "malignant",
        "benign_prob": prob_benign,
        "malignant_prob": 1 - prob_benign,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
