# TITANIC SURVIVAL API

# OVERVIEW
 Built,trained and pipelined a model that predicts whether a person survived or not in tragic titanic accident

## WHAT DID I DO
-I first took the feautures namely Pclass,Age,Sex,Fare as they have much impact on survival probability(our target)
-cleaned the dataset
-Scaled the numeric columns by StandardScaler,str columns by OneHotEncoder,Pclass passes through OrdinalEncoder
-Used Pipeline() to just pass raw data instead of scaling and encoding it every time a new data is passed
-saved it by using joblib to avoid training entire model from scratch 
-connected with Flask and ran a test case

## How TO RUN
-run train-model.py to train the model
-run app.py
-in test.py give the test data and send it to our app 
-run test.py

## SAMPLE REQUEST
    {
    "Pclass":1,
    "Age":29,
    "Fare":100.0,
    "Sex":"female"
    }
    
## SAMPLE RESPONSE
{"prediction": 1}

## TECH USED
-PYTHON
-PANDAS
-FLASK
-SCIKIT-LEARN
-JOBLIB