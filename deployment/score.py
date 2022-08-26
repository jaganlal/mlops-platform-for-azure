import os
import sys
from urllib import response
import numpy as np
import joblib

import math
from azureml.core import Model
from azureml.monitoring import ModelDataCollector
import json
import re
import traceback
import logging
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

'''
Inference script for IRIS Classification:

'''

def init():
    '''
    Initialize required models:
        Get the IRIS Model from Model Registry and load
    '''
    global inputs_dc, prediction_dc
    global model
    global logger

    inputs_dc = ModelDataCollector("simple_iris_model", designation="inputs", feature_names=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
    prediction_dc = ModelDataCollector("simple_iris_model", designation="predictions", feature_names=["Predicted_Species"])

    # model_path = Model.get_model_path(model_name='simple_iris_model:1')

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'models', 'simple_iris_model.pkl')
    logger.info('Model Path:', model_path)
    model = joblib.load(model_path)
    # model = joblib.load(model_path+"/"+"simple_iris_model.pkl")
    logger.info('IRIS model loaded...')

def create_response(predicted_lbl):
    '''
    Create the Response object
    Arguments :
        predicted_label : Predicted IRIS Species
    Returns :
        Response JSON object
    '''
    resp_dict = {}
    logger.info("Predicted Species : ",predicted_lbl)
    resp_dict["predicted_species"] = str(predicted_lbl)
    return json.loads(json.dumps({"output" : resp_dict}))

def run(raw_data):
    '''
    Get the inputs and predict the IRIS Species
    Arguments : 
        raw_data : SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm
    Returns :
        Predicted IRIS Species
    '''
    try:
        logger.info('Request data:', raw_data)
        data = json.loads(raw_data)
        sepal_l_cm = data['SepalLengthCm']
        sepal_w_cm = data['SepalWidthCm']
        petal_l_cm = data['PetalLengthCm']
        petal_w_cm = data['PetalWidthCm']

        # This call is saving our input data into Azure Blob
        inputs_dc.collect([sepal_l_cm,sepal_w_cm,petal_l_cm,petal_w_cm])

        predicted_species = model.predict([[sepal_l_cm,sepal_w_cm,petal_l_cm,petal_w_cm]])[0]

        # This call is saving our prediction data into Azure Blob
        prediction_dc.collect(predicted_species)

        response = create_response(predicted_species)
        logger.info('Response:', response)
        return response

    except Exception as err:
        logger.error(err)
        traceback.print_exc()