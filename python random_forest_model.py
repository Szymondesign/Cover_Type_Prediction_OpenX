#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from OpenX.utils import split_and_scale

file_path = 'covtype.data'
df = pd.read_csv(file_path, header=None, names=column_names)

undersampled_df = df.copy()
class_subsets = [undersampled_df.query("Cover_Type == " + str(i)) for i in range(7)]

for i in range(7):
    class_subsets[i] = class_subsets[i].sample(min_class_size, replace=False, random_state=123)


undersampled_df = pd.concat(class_subsets, axis=0).sample(frac=1.0, random_state=123).reset_index(drop=True)


X_train, X_test, y_train, y_test = split_and_scale(undersampled_df)


model4 = RandomForestClassifier(n_estimators=100, random_state=123)


cv_scores = cross_val_score(model4, X_train, y_train, cv=5, scoring='accuracy')

print("Model 4 cross val:", cv_scores)
print("Model 4 average cross val:", cv_scores.mean())


model4.fit(X_train, y_train)

y_pred = model4.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model 4 accuracy:", accuracy)

