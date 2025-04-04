import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import make_classification, make_regression
from starter.ml.model import inference, train_lr_model








@pytest.fixture
def sample_data():
    """_summary_
    """
    X,y = make_classification(n_samples = 100, n_features=5, random_state=42)
    return X, y


def test_train_lr_model(sample_data):
    X, y = sample_data
    model = train_lr_model(X, y)

    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
