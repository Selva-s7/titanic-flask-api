from flask import Flask,request,jsonify
import joblib
import pandas as pd
pipeline=joblib.load('titanic_pipeline.joblib')
app=Flask(__name__)
@app.route('/health',methods=['GET'])
def disp():
    return jsonify({'status':'ok','model':'titanic-pipeline'})
@app.route('/predict',methods=['POST'])
def predictor():
    data=request.get_json()
    required=['Pclass','Fare','Age','Sex']
    for field in required:
        if field not in data:
            return jsonify({'error':f"{field} is required"}),400
    df=pd.DataFrame([data])
    res=pipeline.predict(df)
    return jsonify({'prediction':int(res[0])})
if __name__=='__main__':
    app.run(debug=True)
    
