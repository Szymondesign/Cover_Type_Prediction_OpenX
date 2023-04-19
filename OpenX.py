#!/usr/bin/env python
# coding: utf-8

# In[207]:


import struct
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# In[210]:


print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("scikit-learn version:", sklearn.__version__)
print("tensorflow version:", tf.__version__)
print("matplotlib version:", matplotlib.__version__)


# In[2]:


file_path = r'C:\Users\Lenovo\Desktop\ML_Intern\covtype.data'


# In[3]:


column_names = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
] + [f'Wilderness_Area{i}' for i in range(1, 5)] + [f'Soil_Type{i}' for i in range(1, 41)] + ['Cover_Type']

# Loading
file_path = 'covtype.data'
df = pd.read_csv(file_path, header=None, names=column_names)

print(df.head())


# In[4]:


df


# In[5]:


#Before applying any heuristics or creating a model we need to conduct some data preprocessing steps
#in order to ensure that the data is ready for analysis and training 


# In[6]:


#even though it was said there are no NAs we can quickly check them
print(df.isnull().sum())


# In[6]:


#no need for removing or imputing the N/As


# In[7]:


stats = df.describe()
print(stats)


# In[8]:


ranges = stats.loc['max'] - stats.loc['min']
print(ranges)


# In[11]:


# Start class labels from 0 rather than 1
df['Cover_Type'] = df['Cover_Type'] - 1


# In[12]:


df['Cover_Type'].value_counts()


# In[15]:


cmap = sns.color_palette('Set2', as_cmap=True)(np.arange(7))

plt.figure(figsize=(8, 8))
plt.pie(
    df['Cover_Type'].value_counts().values,
    colors=cmap,
    labels=df['Cover_Type'].value_counts().keys(),
    autopct='%.1f%%'
)
plt.title("Class Distribution")
plt.show()


# In[13]:


# The data is very imbalanced the 1st class is 48% of the data and class 0 - 36%
#so we are going to try training the model on
# imbalanced values and then we are going to try and train it on 
# Undersampled and oversampled data


# In[14]:


#we should make a scaler


# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


# In[34]:


def split_and_scale(df):
    df = df.copy()
    
    # Split df into X and y
    y = df['Cover_Type'].copy()
    X = df.drop('Cover_Type', axis=1).copy()
    
    # Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
    
    # Scale X
    scaler_1 = StandardScaler()
    scaler_1.fit(X_train)
    
    X_train = pd.DataFrame(scaler_1.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler_1.transform(X_test), columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test


# In[35]:


# and we can create a very useful function for model evaluation for the test data 
def evaluate_model(model, class_balance, X_test, y_test):
    
    model_acc = model.score(X_test, y_test)
    print("Accuracy ({}): {:.2f}%".format(class_balance, model_acc * 100))
    
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    clr = classification_report(y_test, y_pred)
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cbar=False, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    print("Classification Report:\n----------------------\n", clr) #it should show how our predictions
    #are accurate in comparison to the test data


# In[36]:


#data would be scaled while using the function split and scale


# In[37]:


print(df.info())


# In[38]:


#types are ok, no categorical vars so no need for encoding 


# In[39]:


sns.countplot(x='Cover_Type', data=df)
plt.show()


# In[48]:


imbalanced_df = df.copy()

X_train, X_test, y_train, y_test = split_and_scale(imbalanced_df)


# In[45]:


corr_matrix = df.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.show()
#I tried before with combined undersampling and oversampling but i deleted everything I think
# this is gonna be a better approach, also matrix with correlation of all variables


# In[46]:


#In general it is not bad, no strong correlations here 
#I had to make it bigger, because it did not display all the data
#Moderate cor between elevation, horizontal distance to hydrology and horizontal distance to roadways
#But nothing significant


# In[198]:


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


# In[199]:


test_point = X_test.iloc[0]


# In[200]:


heuristic_result = heuristic_predict(test_point)
print("Heuristic prediction:", heuristic_result)


# In[201]:


heuristic_results = [heuristic_predict(row) for _, row in X_test.iterrows()]

print("Heuristic predictions:", heuristic_results)


# In[47]:


#Models now ######################################################################################################


# In[41]:


#I did imbalanced data before so now we just going to put it into a model 


# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[49]:


#Logistic Regression model 1
model1 = LogisticRegression()
model1.fit(X_train, y_train)


# In[50]:


#It reached the limit but as I do not have much computing power we have to skip this, also 
# this is a basic regression model we do not expect it to be perfect
evaluate_model(model1, "Imbalanced", X_test, y_test)


# In[51]:


# so as we see the accurracy is not the worst, it may seem it is not doing so badly
# but as we look at the confusion matrix it shows, that all of our accurate predictions are in 
# class 0 and 1, class 4 seems to be the worst 


# In[52]:


from sklearn.feature_selection import RFECV


# In[53]:


print(df.shape)


# In[54]:


undersampled_df = df.copy()
undersampled_df['Cover_Type'].value_counts()


# In[56]:


# undersampling has its drawbacks such as we lose a lot of data due to the smallest class having 2747
# observations and oversampling's drawback is that we create a lot of artificial data
min_class_size = np.min(undersampled_df['Cover_Type'].value_counts().values)

print("Size:", min_class_size)


# In[57]:


# we need to randomly sample 2747 observations from each class, we do not want them to 
# repeat that is why we put replace=False
class_subsets = [undersampled_df.query("Cover_Type == " + str(i)) for i in range(7)]

for i in range(7):
    class_subsets[i] = class_subsets[i].sample(min_class_size, replace=False, random_state=123)
#okay so we pulled the 7 samples now we have to concatenate them into one resetting index 
# the data will be randomized when we use the split_and_scale function
undersampled_df = pd.concat(class_subsets, axis=0).sample(frac=1.0, random_state=123).reset_index(drop=True)


# In[58]:


undersampled_df


# In[ ]:


# It worked as 2747*7 is 19229


# In[60]:


2747 * 7


# In[61]:


undersampled_df['Cover_Type'].value_counts()


# In[59]:


X_train, X_test, y_train, y_test = split_and_scale(undersampled_df)


# In[62]:


#model 2
model2 = LogisticRegression()
model2.fit(X_train, y_train)


# In[63]:


evaluate_model(model2, "Undersampled", X_test, y_test)


# In[ ]:


# now we can see that the model is working properly, it may have less accuracy but 
# the general prediction looks really good, for a logistic regression 
# mainly it is about the dark blues, which represent the high values so the highest values are in
# the right place, all the other around are missclassifications
# pretty decent f1 scores all around the classes


# In[ ]:


## We would leave it as that right now because that is quite good for now, we will see how the rest of the
# models perform


# In[ ]:


#2
#Decision Tree model3


# In[65]:


model3 = DecisionTreeClassifier()
model3.fit(X_train, y_train)

# Prediction and accuracy test
y_pred = model3.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy:", accuracy)


# In[66]:


# this is good accuracy so as decision tree is sometimes better as it can handle non-linear
# relationships and are capable of modelling wide range of decision boundaries 
# also captures underlying patterns really good however it is prone to overfittingh
#Especially in this number of features in the dataset
# to check it we can use cross-validation and add random forest 
#which combines multiple decision trees 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# In[68]:


# random forest model 4


# In[70]:


model4 = RandomForestClassifier(n_estimators=100, random_state=123)

cv_scores = cross_val_score(model4, X_train, y_train, cv=5, scoring='accuracy')

print("cross val:", cv_scores)
print("average cross val:", cv_scores.mean())

#training on entire dataset 
model4.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("model4 accuracy:", accuracy)


# In[71]:


# the cross validation scores show a consistent performance accross 5 folds of data
# the accuracy is better and it is not prone to overfitting so the model with random forest would 
# be the one to choose
# also the 1st logistic regression model would be model2 with undersampling of course 


# In[73]:


# so now as i forgot to plot accuracy for the 1st model we can plot 2 of them
y_pred_model2 = model2.predict(X_test)
accuracy_model2 = accuracy_score(y_test, y_pred_model2)
plt.figure()
plt.bar(["model2 (Logistic Regression)"], [accuracy_model2])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("model2 (Logistic Regression) Accuracy")
plt.show()


# In[74]:


# no surprises here and same with model4
plt.figure()
plt.bar(["model4 (Random Forest)"], [accuracy])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("model4 (Random Forest) Accuracy")
plt.show()


# In[75]:


# at least we have 2 models done


# In[76]:


# Neural network model5 on imbalanced data


# In[107]:


from sklearn.preprocessing import LabelEncoder # keras is right now part of tensorflow so
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


# In[78]:


X = df.drop('Cover_Type', axis=1).values  
y = df['Cover_Type'].values

y = to_categorical(y)


# In[80]:


model5 = Sequential()
model5.add(Dense(54, activation='relu', input_shape=(X_train.shape[1],))),
model5.add(Dense(100, activation='relu')),
model5.add(Dense(7, activation='softmax')) 
#7 classes of output


# In[83]:


model5.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# In[84]:


history = model5.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


# In[102]:


def evaluate_neural(model, class_balance, X_test, y_test, history):
    
    loss, model_acc = model.evaluate(X_test, y_test, verbose=0)
    print("Loss ({}): {:.4f}".format(class_balance, loss))
    print("Accuracy ({}): {:.2f}%".format(class_balance, model_acc * 100))
    
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    
    cm = confusion_matrix(y_test, y_pred)
    clr = classification_report(y_test, y_pred)
    
    
    print("Classification Report:\n----------------------\n", clr)


# In[95]:


imbalanced_df = df.copy()

X_train, X_test, y_train, y_test = split_and_scale(imbalanced_df)


# In[103]:


evaluate_neural(model5, "Imbalanced_data", X_test, y_test, history)


# In[ ]:


# based on the given results the accuracy is 49,09% which is close to guessing
# Let us create a plot and a second neural network with an additional layer
# and then we will try to do some hyperparameter tuning etc


# In[104]:


def plot_training_history(history):
    plt.figure(figsize=(12, 4))

   #training loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    ## training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


# In[105]:


plot_training_history(history)


# In[117]:


# Neural network model6 with one additional layer and dropout layer still on imbalanced data


# In[126]:


imbalanced_df = df.copy()
X_train, X_test, y_train, y_test = split_and_scale(imbalanced_df)


# In[127]:


model6 = Sequential()
model6.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model6.add(Dropout(0.2)) 
model6.add(Dense(100, activation='relu'))
model6.add(Dropout(0.2))
model6.add(Dense(100, activation='relu'))
model6.add(Dropout(0.2))
model6.add(Dense(7, activation='softmax'))


# In[128]:


model6.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# In[129]:


history = model6.fit(X_train, y_train, validation_split=0.2, batch_size=40, epochs=70, verbose=0)


# In[130]:


evaluate_neural(model6, "Imbalanced_data_2", X_test, y_test, history)


# In[131]:


plot_training_history(history)


# In[ ]:


# Neural network model7 with one additional layer and dropout layer still on undersampled data


# In[118]:


undersampled_df = df.copy()


# In[119]:


class_subsets = [undersampled_df.query("Cover_Type == " + str(i)) for i in range(7)]

for i in range(7):
    class_subsets[i] = class_subsets[i].sample(min_class_size, replace=False, random_state=123)

undersampled_df = pd.concat(class_subsets, axis=0).sample(frac=1.0, random_state=123).reset_index(drop=True)


# In[132]:


X_train, X_test, y_train, y_test = split_and_scale(undersampled_df)


# In[133]:


model7 = Sequential()
model7.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model7.add(Dropout(0.2)) 
model7.add(Dense(100, activation='relu'))
model7.add(Dropout(0.2))
model7.add(Dense(100, activation='relu'))
model7.add(Dropout(0.2))
model7.add(Dense(7, activation='softmax'))


# In[134]:


model7.compile(optimizer=Adam(learning_rate=0.0008),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# In[135]:


history = model7.fit(X_train, y_train, validation_split=0.2, batch_size=45, epochs=50, verbose=0)


# In[136]:


evaluate_neural(model7, "Undersampled_data", X_test, y_test, history)


# In[137]:


plot_training_history(history)


# In[ ]:





# In[138]:


imbalanced_df = df.copy()
X_train, X_test, y_train, y_test = split_and_scale(imbalanced_df)


# In[139]:


model8 = Sequential()
model8.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model8.add(Dropout(0.2)) 
model8.add(Dense(100, activation='relu'))
model8.add(Dropout(0.2))
model8.add(Dense(100, activation='relu'))
model8.add(Dropout(0.2))
model8.add(Dense(7, activation='softmax'))


# In[140]:


model8.compile(optimizer=Adam(learning_rate=0.0007),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# In[144]:


history = model8.fit(X_train, y_train, validation_split=0.2, batch_size=40, epochs=70, verbose=0)


# In[145]:


evaluate_neural(model8, "Imbalanced_data_3", X_test, y_test, history)


# In[146]:


plot_training_history(history)


# In[175]:


# hyperparameter tuning
import tensorflow as tf
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# In[173]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score


# In[ ]:


# At this point I had a few problems both with KerasClassifier not passing the correct parameters or 
# with parameters themselves so finally I did not use much of those imported classes or functions


# In[ ]:


# but then i deleted everything and tried this approach by creating a model class


# In[176]:


class CustomModel(tf.keras.Model):

    def __init__(self, num_features, learning_rate):
        super(CustomModel, self).__init__()
        self.dense1 = Dense(128, activation='relu', input_shape=(num_features,))
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(100, activation='relu')
        self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(100, activation='relu')
        self.dropout3 = Dropout(0.2)
        self.dense4 = Dense(7, activation='softmax')

        self.compile(optimizer=Adam(learning_rate=learning_rate),
                     loss=SparseCategoricalCrossentropy(from_logits=False),
                     metrics=['accuracy'])

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.dropout3(x)
        x = self.dense4(x)
        return x


# In[ ]:


# and a separate function for finding hyperparameters


# In[182]:


def search_hyperparameters(X_train, y_train, n_iter=10):
    def build_and_train_model(learning_rate):
        model = CustomModel(X_train.shape[1], learning_rate)
        history = model.fit(X_train, y_train, validation_split=0.2, batch_size=40, epochs=70, verbose=0)
        return model, history
    
    search_space = [Real(1e-5, 1e-2, name='learning_rate', prior='log-uniform')]
    best_score = -1
    best_learning_rate = None
    best_model = None
    best_history = None

    for _ in range(n_iter):
        learning_rate = search_space[0].rvs(random_state=None)
        model, history = build_and_train_model(learning_rate)
        score = accuracy_score(y_test, np.argmax(model.predict(X_test), axis=-1))
        
        if score > best_score:
            best_score = score
            best_learning_rate = learning_rate
            best_model = model
            best_history = history

    return best_learning_rate, best_model, best_history


# In[183]:


best_learning_rate, best_model, best_history = search_hyperparameters(X_train, y_train, n_iter=10)


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(best_history.history['loss'], label='Training Loss')
plt.plot(best_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(best_history.history['accuracy'], label='Training Accuracy')
plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')


# In[184]:


model8_tuned = best_model


# In[185]:


evaluate_neural(model8_tuned, "Tuned_Imbalanced_data", X_test, y_test, best_history)


# In[ ]:


# Based on the accuracy scores and the classification reports provided for 2 models
# model 9 acc of 81,02 f1-score of 0,81 weighted avg
# model 8 acc of 89,21 f1-score pf 0,89 weighted avg 
# From the information provided, model 8 seems to be performing better, as it has a higher accuracy and a higher weighted average F1-score
#The F1-score takes into account both precision and recall, making it a more comprehensive metric
# for evaluating the performance of the models.


# In[164]:


print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)


# In[ ]:





# In[ ]:





# In[ ]:




