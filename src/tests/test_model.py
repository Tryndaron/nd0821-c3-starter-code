import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import make_classification, make_regression
from starter.ml.model import inference, train_lr_model, compute_model_metrics
from starter.ml.data import process_data
from unittest.mock import Mock, patch
from starter.train_model import cat_features






@pytest.fixture
def sample_data():
    """_summary_
    """
    X,y = make_classification(n_samples = 100, n_features=5, random_state=42)
    return X, y

def test_compute_model_metrics():
    data = pd.read_csv("src/data/census.csv")
    train, test = train_test_split(data, test_size=0.30 , stratify=data['salary']  )

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True)
    
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb)
    
    lr_model= train_lr_model(X_train, y_train)
    pred= inference(lr_model, X_test)
    precision, recall, fbeta = compute_model_metrics(
        y_test, pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)





def test_train_lr_model(sample_data):
    X, y = sample_data
    model = train_lr_model(X, y)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")



def test_inference():
    model_lr = Mock()
    X_train = Mock()
    prediction = inference(model_lr, X_train)
    assert prediction is not None
    model_lr.predict.assert_called_with(X_train)