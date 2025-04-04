from fastapi.testclient import TestClient
from main import app



client = TestClient(app)


def test_root_message():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == "This is my first API!"


def test_lower_50():
    census1={
        'age': 39,
        'workclass': 'State-gov',
        'fnlgt': '77516',
        'education': 'Bachelors',
        'education_num': 13,
        'marital-status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationsship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 2174,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States',
    }
    resp = client.post("/inference", json=census1)
    assert resp.status_code == 200
    assert resp.json() == {'predicted_salary': '<=50k'}  

def test_higher_50():
    census2 ={
        'age': 43,
        'workclass': 'Self-emp-not-inc',
        'fnlgt': '292175',
        'education': 'Masters',
        'education_num': 14 ,
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
    resp = client.post("/inference", json=census2)
    assert resp.status_code == 200
    assert resp.json() == {'predicted_salary': '>50k'} 