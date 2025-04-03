import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import make_classification, make_regression
from model import inference, train_model

@pytest.fixture
def classification_data():
    """Erstell Beispiel Daten für classification
    """
    X_train, y_train = make_classification(n_samples=100, n_features=5, random_state=42)
    X_test, _ = make_classification(n_samples=10, n_features=5, random_state=24)
    return X_train, y_train, X_test

@pytest.fixture
def regression_data():
    X_train, y_train = make_regression(n_samples=100, n_features=5, random_state=42)
    X_test, _ = make_regression(n_samples=10, n_features=5, random_state=24)
    return X_train, y_train, X_test


def test_train_classification(classification_data):
    X_train, y_train = classification_data
    model = train_model(X_train, y_train, model_type="classifier")
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")


def test_train_regression(regression_data):
    X_train, y_train = regression_data
    model = train_model(X_train, y_train, model_type="classifier")
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")


def test_inference_classification(classification_data):
    X_train, y_train, X_test = classification_data
    model = train_model(X_train, y_train, model_type="classifier")
    predictions = inference(model, X_test)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test)

def test_inference_regression(regression_data):
    X_train, y_train, X_test = regression_data
    model = train_model(X_train, y_train, model_type="regressor")
    predictions = inference(model, X_test)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test)