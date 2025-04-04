# Put the code for your API here.
from fastapi import FastAPI
import pandas as pd
import sklearn.preprocessing
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.train_model import cat_features
from starter.ml.model import inference
from joblib import load

lr_model = load('starter/model/lr_model.joblib')
encoder = load('starter/model/encoder_path.joblib')
lb = load('starter/model/lb_path.joblib')


class Census_Data(BaseModel):
    age: float
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias='education-num')
    marital_status: str = Field(...,alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias='capital-gain')
    capital_loss: int = Field(..., alias='capital-loss')
    hours_per_week: int = Field(..., alias='hours-per-week')
    native_country: str = Field(..., alias='nativ-country')




app = FastAPI()


@app.get("/")
async def root():
    return "This is my first API!" 


@app.post("/inference")
async def model_inference(census: Census_Data):
    Census_Data_df = pd.DataFrame(census.dict(by_alias=True), index=[0] )
    input_data, _, _, _ = process_data(Census_Data_df,  cat_features, label=None
                                       , training=False, encoder=encoder, lb=lb)
    
    pred = inference(lr_model, input_data)
    pred_class = lb.inverse_transform(pred)[0]
    resp = {'predicted_salary': pred_class}
    return resp  











