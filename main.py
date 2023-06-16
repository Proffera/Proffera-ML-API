import os
from typing import Union
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import uvicorn

PORT = int(os.environ.get("PORT", 8080))

app = FastAPI()

# Load the model
model = tf.keras.models.load_model("model_v2.h5")



@app.get("/")
def read_root():
    return {"Hello": "World"}

class InputData(BaseModel):
    Kategori: str
    Sumber_Dana: str = Field(..., alias="Sumber Dana")
    Nama_KLDP: str = Field(..., alias="Nama KLDP")
    HPS: float
    PAGU: float
    Satker: str

@app.post("/predict")
def predict(input_data: InputData):
    input_dict = input_data.dict()
    input_dict["Sumber Dana"] = input_dict.pop("Sumber_Dana")
    input_dict["Nama KLDP"] = input_dict.pop("Nama_KLDP")

    df = pd.DataFrame([input_dict])

    # Label encoding for categorical variables
    categorical_columns = ['Kategori', 'Sumber Dana', 'Nama KLDP', 'Satker']
    label_encoders = {}

    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    
    # Normalize numerical features
    numerical_columns = ['HPS', 'PAGU']
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Make prediction
    prediction = float(model.predict(df)[0][0])

    # Return the prediction as the response
    return {"Predicted Score": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
