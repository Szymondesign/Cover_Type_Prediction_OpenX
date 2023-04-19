#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from OpenX.utils import split_and_scale

# Load the preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# Undersample the majority class
min_class_size = df['Cover_Type'].value_counts().min()
undersampled_df = df.copy()
class_subsets = [undersampled_df.query("Cover_Type == " + str(i)) for i in range(7)]

for i in range(7):
    class_subsets[i] = class_subsets[i].sample(min_class_size, replace=False, random_state=123)

undersampled_df = pd.concat(class_subsets, axis=0).sample(frac=1.0, random_state=123).reset_index(drop=True)

# Split and scale the data
X_train, X_test, y_train, y_test = split_and_scale(undersampled_df)

# Train the Logistic Regression model
model2 = LogisticRegression()
model2.fit(X_train, y_train)

# Evaluate the model
y_pred = model2.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion Matrix:\n', confusion_mat)
print('Classification Report:\n', class_report)

