from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import xgboost as xgb
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
booster = xgb.Booster()
booster.load_model("model/model.json")

@app.get("/predict")
def predict(s1: float, s2: float, s3: float, s4: float, s5: float, s6: float):
    X = np.array([[s1, s2, s3, s4, s5, s6]])
    dmat = xgb.DMatrix(X)
    pred = booster.predict(dmat)
    return {"prediction": float(pred[0])}
