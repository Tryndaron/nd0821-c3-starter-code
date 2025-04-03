from fastapi.testclient import TestClient
from ..starter.main import app

client = TestClient(app)


def test_root_message():
    resp = client.get('/')
    assert resp.status_code == 200
    assert resp.json() == 'message: Welcome to my first API ! You can get an inference from a machine learning model here'


def test_lower_50():
    census={
        'age': 40,
        'workclass': 'State-gov',
        'fnlgt': '77516',
        'education': 'Bachelors',
        'marital-status': 'Never-Married',
        'occupation': 'Adm-clerical',
        'relationsship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 2174,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States',
    }
    resp = client.post('/inference', json=census)
    assert resp.status_code == 200
    assert resp.json() == {'predicted_salary': '<=50k'}  

def test_higher_50():
    census ={
        'age': 43,
        'workclass': 'Self-emp-not-inc',
        'fnlgt': '292175',
        'education': 'Masters',
        'marital-status': 'Divorced',
        'occupation': 'Exec-managerial',
        'relationsship': 'Unmarried',
        'race': 'White',
        'sex': 'Female',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 45,
        'native-country': 'United-States',
    }
    resp = client.post('/inference', json=census)
    assert resp.status_code == 200
    assert resp.json() == {'predicted_salary': '>50k'} 