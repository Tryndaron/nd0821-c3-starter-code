
import requests
import json

# Define the URL for the FastAPI endpoint

url = "http://127.0.0.1:8000/predict"

# Sample data for inference
person = {""
        "age": 43,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 292175,
        "education": "Masters",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-states"
        ""
        }
    
# Make the POST request to the inference endpoint
response = requests.post(url, json=person)
if response.status_code == 200:
    # Print the predictions
    print("Predictions:", response.json())
else:
    print(f"Request failed with status code {response.json()}")