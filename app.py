from flask import Flask,request,jsonify
import joblib
import pandas as pd
pipeline=joblib.load('titanic_pipeline.joblib')
app=Flask(__name__)
@app.route('/predict',methods=['POST'])
def predictor():
    data=request.get_json()
    df=pd.DataFrame([data])
    res=pipeline.predict(df)
    return jsonify({'prediction':int(res[0])})
if __name__=='__main__':
    app.run(debug=True)
    
