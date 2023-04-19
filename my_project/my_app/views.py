from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import model2, model4, heuristic_predict, model8
import numpy as np

@api_view(['POST'])
def predict_cover_type(request):
    # Extract input data from the request
    input_data = request.data

    # Call the appropriate model based on the input data
    model_name = input_data.get('model')
    features = input_data.get('data')

    if model_name == 'heuristics':
        prediction = heuristic_predict(features)
    elif model_name == 'model2':
        input_array = np.array([features])
        prediction = model2.predict(input_array)
    elif model_name == 'model4':
        input_array = np.array([features])
        prediction = model4.predict(input_array)
    elif model_name == 'model8':
        input_array = np.array([features])
        prediction = np.argmax(model8.predict(input_array), axis=-1)
    else:
        return Response({"error": "Invalid model specified."})

    # Return the prediction as a JSON response
    return Response({"prediction": int(prediction[0])})