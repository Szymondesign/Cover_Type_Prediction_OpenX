# Cover_Type_Prediction_OpenX
The purpose of this project is to predict forest cover types based on cartographic variables. The models are trained on the Cover_Type dataset, which consists of 581,012 instances and 54 attributes. The target variable, 'Cover_Type', has 7 different classes representing forest cover types.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results and Comparison](#results-and-comparison)
6. [REST API](#REST-API)
7. [License](#license)

## Introduction

This project aims to predict forest cover types based on cartographic variables using the Covertype dataset. The dataset consists of 581,012 instances and 54 attributes, with the target variable 'Cover_Type' having 7 different classes representing forest cover types.

The project involves implementing and comparing different models, including a simple heuristic, two baseline machine learning models using Scikit-learn, and a neural network using TensorFlow. The TensorFlow neural network will include a function to find the best set of hyperparameters, and training curves for the best hyperparameters will be plotted.

Models will be evaluated using appropriate plots and/or metrics to compare their performance. The project also includes the development of a simple REST API that serves the models, allowing users to choose a model (heuristic, two baseline models, or neural network) and input necessary features to receive a prediction. While the API will not be hosted, the code will be provided.

## Requirements

- Python 3.6+
- pandas version: 1.4.4
- numpy version: 1.21.5
- scikit-learn version: 1.0.2
- tensorflow version: 2.11.0
- matplotlib version: 3.5.2
- django
- djangorestframework


## Installation

1. Clone the repository:
git clone (https://github.com/Szymondesign/Cover_Type_Prediction_OpenX.git)


2. Install the required packages:
pip install -r requirements.txt



## Usage

To run the models, navigate to the project directory and execute the Python scripts for each model:

- Heuristics: `python heuristics_model.py`
- Logistic Regression: `python logistic_regression_model.py`
- Random Forest: `python random_forest_model.py`
- Neural Network: `python neural_network_model.py`

## Results and Comparison

Based on the evaluation metrics, the best-performing model is the neural network model (Model 8), with an accuracy of 89.21% on the test set. The Random Forest model (Model 4) also performed well, with an average cross-validation accuracy of 85.93% and a test accuracy of 86.86%. The Logistic Regression model (Model 3) had the lowest accuracy among the three models, with an accuracy of 71.00%.

One drawback of the neural network model is that it requires a lot of computational resources and time to train, especially if the dataset is large. On the other hand, the Random Forest model can be trained relatively quickly and can handle high-dimensional data, but it may not perform well if there are complex relationships between the features and the target variable. The Logistic Regression model is simple and easy to interpret, but it may not be suitable for datasets with non-linear relationships between the features and the target variable.

In terms of the dataset used, Model 8 was trained on an imbalanced dataset, which can lead to biased results and poor generalization to new data. To address this issue, undersampling was used to balance the class distribution in Model 4. However, Model 3 was trained on the original imbalanced dataset without any balancing techniques.

Overall, each model has its own strengths and weaknesses, and the choice of model should depend on the specific problem and the characteristics of the dataset. In this case, Model 8 (neural network) and Model 4 (Random Forest) performed well and could be good candidates for further testing and deployment.

Quick note about the hyperparameter tuned model and normal NN model: 
Based on the accuracy scores and the classification reports provided for both models, here's a comparison:

Model 9: Tuned Imbalanced Data

Accuracy: 81.02%
F1-score (weighted avg): 0.81
Model 8: Imbalanced Data 3

Accuracy: 89.21%
F1-score (weighted avg): 0.89
From the information provided, Model 8 (Imbalanced Data 3) seems to be performing better, as it has a higher accuracy and a higher weighted average F1-score. The F1-score takes into account both precision and recall, making it a more comprehensive metric for evaluating the performance of the models.

## REST API

REST API to predict the forest cover type using the available models. Located in my_project file. Send a POST request to the following endpoint with the input data as JSON:

Endpoint: `/api/predict_cover_type/`

Example request:

```json
{
    "model": "heuristics",
    "data": {
        "attribute1": 1,
        "attribute2": 2,
        ...
    }
}

The response will include the prediction as JSON:
{
    "prediction": "Cover_Type_1"
}

 ## License

This project is licensed under the [MIT License](LICENSE).
