#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from OpenX.utils import split_and_scale


imbalanced_df = pd.read_csv('covtype.data')


X_train, X_test, y_train, y_test = split_and_scale(imbalanced_df)


model8 = Sequential()
model8.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model8.add(Dropout(0.2))
model8.add(Dense(100, activation='relu'))
model8.add(Dropout(0.2))
model8.add(Dense(100, activation='relu'))
model8.add(Dropout(0.2))
model8.add(Dense(7, activation='softmax'))

model8.compile(optimizer=Adam(learning_rate=0.0007),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


history = model8.fit(X_train, y_train, validation_split=0.2, batch_size=40, epochs=70, verbose=0)


test_loss, test_accuracy = model8.evaluate(X_test, y_test, verbose=0)
print("Model 8 test loss:", test_loss)
print("Model 8 test accuracy:", test_accuracy)

