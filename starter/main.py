# Put the code for your API here.
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from starter.ml.data import process_data
from starter.train_model import cat_features
from joblib import load

lr_model = load('./model/lr_model.joblib')
encoder = load('./model/encoder_path.joblib')
lb = load('./model/lb_path.joblib')


class Census_Data(BaseModel):
    age: float
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str




app = FastAPI()


@app.get("/")
async def root():
    return {"message: Welcome to my first API ! You can get an inference from"
    " a machine learning model here"} 


@app.post("/inference")
async def model_inference(census: Census_Data):
    Census_Data_df = pd.DataFrame(census.dict(by_alias=True), index=[0] )
    input_data, _, _, _ = process_data(Census_Data_df,  cat_features, label=None
                                       , training=False, encoder=encoder, lb=lb)
    
    pred = lr_model.inference(lr_model, input_data)
    pred_class = lb.inverse_transform(pred)[0]
    resp ={'predicted_salary': pred_class}
    return resp  











