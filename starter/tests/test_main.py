from fastapi.testclient import TestClient
from main import app



client = TestClient(app)


def test_root_message():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == "This is my first API!"




def test_lower_50k():
    person = {""
    "age": 0,
    "workclass": "string",
    "fnlgt": 0,
    "education": "string",
    "education-num": 0,
    "marital-status": "string",
    "occupation": "string",
    "relationship": "string",
    "race": "string",
    "sex": "string",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 0,
    "native-country": "string"
    ""
    }
    resp = client.post("/predict", json=person)
    assert resp.status_code == 200
    assert resp.json() == {"predictions":["<=50K"] }  


def test_higher_50k():
    person = {""
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
    'native-country': 'United-States'
    ""
    }
    resp = client.post("/predict", json=person)
    assert resp.status_code == 200
    assert resp.json() == {"predictions":[">50K"] }




""" def test_higher_50():
    census2 = {'age': 43,
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
        'native-country': 'United-States'}
    resp = client.post("/inference", json=census2)
    assert resp.status_code == 200
    assert resp.json() == {'predicted_salary': '>50K'}  """