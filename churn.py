import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import joblib


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 
from sklearn import tree
import pydotplus


""" Here lies the code for the churn analysis. To add it to django, joblib tool is used.
        No need to create the classifier in django"""




data = pd.read_csv("user_satisfaction_survey.csv")
#data.head()

X = data.drop(['Churn','ID'], axis=1) #Dropping the ID column also - its not needed
y = data['Churn']

"""OneHotEncoder will be used to turn the categorical (yes, no) to numerical"""
dtree = DecisionTreeClassifier()
ohe = OneHotEncoder()

"""Column_transformer will be used to apply the OneHotEncoder to all the columns except Classes_per_week (that's already numerical)"""
column_trans = make_column_transformer((OneHotEncoder(),['Happy_with_instructors', 'Happy_with_class_duration', 'Happy_with_class_timings', \
    'Happy_with_class_size', 'Happy_with_facilities', 'Happy_with_price']),remainder='passthrough')

"""Use pipeline to chain steps together. Applies 'column_transform' to data and passes it through decision tree classifier
    Apply 5-fold cross validation to pipeline and getting the mean accuracy"""

pipe = make_pipeline(column_trans, dtree)

# Mean refers how well trained the classifier is. Good amount and diverese data makes the classifier better.
mean = cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
print(mean)

# Train our pipleline with the data
pipe.fit(X,y)
# Use a sample from the data and apply our model on it (test with an expected result)
X_new = X.sample(1)

print("\n")
print("X_new churn prediction is: " + pipe.predict(X_new))

# Create new samples and apply the chunr annalysis to them
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
