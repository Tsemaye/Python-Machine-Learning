from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Union, List
import joblib
import pandas as pd
import os

# Defining the path where my models are stored
models_path = "/Users/charisoneyemi/Downloads/611Assignment/Models"

# Loading the trained models using joblib
models = {
    "SVM": joblib.load(os.path.join(models_path, "svm_model.pkl")),
    "KNN": joblib.load(os.path.join(models_path, "knn_model.pkl")),
    "NaiveBayes": joblib.load(os.path.join(models_path, "nb_model.pkl")),
    "MLP": joblib.load(os.path.join(models_path, "mlp_model.pkl")),
    "XGBoost": joblib.load(os.path.join(models_path, "xgb_model.pkl"))
}

app = FastAPI()

class SampleData(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]

@app.post("/predict")
def predict(sample_data: SampleData):
    try:
        sample = sample_data.data
        df = pd.DataFrame(sample if isinstance(sample, list) else [sample])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    predictions = {}
    for name, model in models.items():
        preds = model.predict(df)
        predictions[name] = preds.tolist()

    return {"predictions": predictions}
