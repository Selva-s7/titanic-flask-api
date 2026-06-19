import requests
data={
    "Pclass":1,
    "Age":29,
    "Fare":100.0,
    "Sex":"female"
    }
response=requests.post("http://127.0.0.1:5000/predict",json=data)
print(response.json())