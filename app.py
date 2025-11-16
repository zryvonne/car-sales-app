from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import xgboost as xgb
import numpy as np
from fastapi.responses import FileResponse

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

# Serve frontend
@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

@app.get("/predict")
def predict(s1: float, s2: float, s3: float, s4: float, s5: float, s6: float):
    import pandas as pd

    # Create a DataFrame with correct column names
    df = pd.DataFrame(
        [[s1, s2, s3, s4, s5, s6]],
        columns=["lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6"]
    )

    # Convert to DMatrix
    dmat = xgb.DMatrix(df)

    # Make prediction
    pred = booster.predict(dmat)
    return {"prediction": float(pred[0])}
