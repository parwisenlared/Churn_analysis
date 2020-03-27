import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import joblib

data = pd.read_csv("user_satisfaction_survey.csv")

X = data.drop(['Churn','ID'], axis=1) #Dropping the ID column also - its not needed
y = data['Churn']

dtree = DecisionTreeClassifier()
ohe = OneHotEncoder()

column_trans = make_column_transformer((OneHotEncoder(),['Happy_with_instructors', 'Happy_with_class_duration', 'Happy_with_class_timings', \
    'Happy_with_class_size', 'Happy_with_facilities', 'Happy_with_price']),remainder='passthrough')

pipe = make_pipeline(column_trans, dtree)

cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()

pipe.fit(X,y)

X_new = X.sample(1)

print("\n")
print("X_new churn prediction is: " + pipe.predict(X_new))

sample1 = [3,'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No']
sample1 = pd.DataFrame([sample1], columns = X.columns)
print("\n")
print("Sample1 churn prediction is: " + pipe.predict(sample1))

sample2 = [2,'No', 'Yes', 'No', 'No', 'Yes', 'No']
sample2 = pd.DataFrame([sample2], columns = X.columns)
print("\n")
print("Sample2 churn prediction is: " + pipe.predict(sample2))


joblib.dump(pipe,"dtree_classifier")
dt = joblib.load("dtree_classifier")
print("\n")
print("Joblib prediction for sample 2: " + dt.predict(sample2))
print("\n")