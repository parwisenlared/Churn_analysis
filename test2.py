import pandas as pd
import joblib


cols = ['Classes_per_week','Happy_with_instructors', 
'Happy_with_class_duration','Happy_with_class_timings', 'Happy_with_class_size',
'Happy_with_facilities', 'Happy_with_price']

sample2 = [2,'No', 'Yes', 'No', 'No', 'Yes', 'No']
sample2 = pd.DataFrame([sample2], columns = cols)


dt = joblib.load("dtree_classifier2")

print(dt.predict(sample2))