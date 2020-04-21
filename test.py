import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import joblib


data = pd.read_csv("user_satisfaction_survey.csv")

X = data.drop(['Churn','ID'], axis=1) #Dropping the ID column also - its not needed
y = data['Churn']

dtree = DecisionTreeClassifier()
ohe = OneHotEncoder()

ct = ColumnTransformer(transformers=[("oh",OneHotEncoder(),[1,2,3,4,5,6])], remainder="passthrough")

pipe = make_pipeline(ct, dtree)

print(cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean())

pipe.fit(X,y)

sample1 = [10,'No', 'No', 'No', 'Yes', 'No', 'No']
sample1 = pd.DataFrame([sample1], columns = X.columns)

print(pipe.predict(sample1))

joblib.dump(pipe,"dtree_classifier2")






