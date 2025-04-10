# Put the code for your API here.
from fastapi import FastAPI
import pandas as pd
import sklearn.preprocessing
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.train_model import cat_features
from starter.ml.model import inference
from joblib import load
import pickle




""" lr_model = load('starter/model/lr_model.joblib')
encoder = load('starter/model/encoder_path.joblib')
lb = load('starter/model/lb_path.joblib') """


class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias='education-num')
    marital_status: str = Field(..., alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias='capital-gain')
    capital_loss: int = Field(..., alias='capital-loss')
    hours_per_week: int = Field(..., alias='hours-per-week')
    native_country: str = Field(..., alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                'age': 39,
                'workclass': 'State-gov',
                'fnlgt': 77516,
                'education': 'Bachelors',
                'education-num': 13,
                'marital-status': 'Never-married',
                'occupation': 'Adm-clerical',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 2174,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'United-States'
            }
        }




app = FastAPI()



# Load model and label binarizer
model_path = "src/model/lr_model.pkl"
lb_path = "src/model/lb_path.pkl"

with open(model_path, "rb") as model_file:
    lr_model = pickle.load(model_file)

with open(lb_path, "rb") as lb_file:
    lb = pickle.load(lb_file)

with open("src/model/encoder_path.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)






@app.get("/")
async def root():
    return "This is my first API!" 


@app.post("/predict")
def predict(data:Person):
    # Convert list of DataInput objects to DataFrame
    """ input_df = pd.DataFrame([{"age": data.age,
                        "workclass": data.workclass,
                        "fnlgt": data.fnlgt,
                        "education": data.education,
                        "education-num": data.education_num,
                        "marital-status": data.marital_status,
                        "occupation": data.occupation,
                        "relationship": data.relationship,
                        "race": data.race,
                        "sex": data.sex,
                        "capital-gain": data.capital_gain,
                        "capital-loss": data.capital_loss,
                        "hours-per-week": data.hours_per_week,
                        "native-country": data.native_country}]) """
    person_df = pd.DataFrame(data.dict(by_alias=True), index=[0])

    # Load model and label binarizer
    model_path = "src/model/lr_model.pkl"
    lb_path = "src/model/lb_path.pkl"

    with open(model_path, "rb") as model_file:
        lr_model = pickle.load(model_file)

    with open(lb_path, "rb") as lb_file:
        lb = pickle.load(lb_file)

    with open("src/model/encoder_path.pkl", "rb") as encoder_file:
        encoder = pickle.load(encoder_file)

    # Process input data
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, _, _, _ = process_data(
        person_df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
    )

    # Perform inference
    predictions = inference(lr_model, X)
    # Convert predictions back to original labels
    preds = lb.inverse_transform(predictions)[0] 

    # Return predictions
    return {"predictions": preds.tolist()}






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    




