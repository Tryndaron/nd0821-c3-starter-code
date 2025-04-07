
import requests
import json

# Define the URL for the FastAPI endpoint

url = "http://127.0.0.1:8000/predict"

# Sample data for inference
data = {""
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
    
# Make the POST request to the inference endpoint
response = requests.post(url, json=data)
if response.status_code == 200:
    # Print the predictions
    print("Predictions:", response.json())
else:
    print(f"Request failed with status code {response.json()}")