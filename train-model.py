import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
data=pd.read_csv('train.csv')
data=data.dropna(subset=['Age'])
X=data[['Pclass','Fare','Age','Sex']]
Y=data['Survived']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
numeric_cols=['Fare','Age']
char_cols=['Sex']
ordered_cols=['Pclass']
scaler=StandardScaler()
encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
order=OrdinalEncoder(categories=[[1,2,3]])
preprocessing=ColumnTransformer(transformers=[
    ('num',scaler,numeric_cols),
    ('char',encoder,char_cols),
    ('Order',order,ordered_cols)
])
pipeline=Pipeline(steps=[
    ('process',preprocessing),
    ('model',LogisticRegression())
])
pipeline.fit(X_train,Y_train)
print('ACCURACY OF MODEL IS:',pipeline.score(X_test,Y_test))
joblib.dump(pipeline,'titanic_pipeline.joblib')
