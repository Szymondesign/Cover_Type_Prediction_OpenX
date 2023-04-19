#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from OpenX.utils import split_and_scale


df = pd.read_csv("covtype.data")  

imbalanced_df = df.copy()
X_train, X_test, y_train, y_test = split_and_scale(imbalanced_df)

def heuristic_predict(imbalanced_df):
    if isinstance(imbalanced_df, pd.Series):
        imbalanced_df = imbalanced_df.to_dict()

    elevation = imbalanced_df['Elevation']
    wilderness_area4 = imbalanced_df['Wilderness_Area4']
    soil_type10 = imbalanced_df['Soil_Type10']

    if elevation > 3000 and wilderness_area4 == 1:
        prediction = 1  # Cover_Type 1
    elif elevation > 2000 and soil_type10 == 1:
        prediction = 2  # Cover_Type 2
    else:
        prediction = 3  # Cover_Type 3

    return prediction

heuristic_results = [heuristic_predict(row) for _, row in X_test.iterrows()]
print("Heuristic predictions:", heuristic_results)

