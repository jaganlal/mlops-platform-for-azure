import os
import sys
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

'''
Inference script for IRIS Classification:

'''

def init():
    '''
    Initialize required models:
        Get the IRIS Model from Model Registry and load
    '''
    global prediction_dc
    global model
    logging.setLevel(logging.DEBUG)

    prediction_dc = ModelDataCollector("IRIS", designation="predictions", feature_names=["SepalLengthCm","SepalWidthCm", "PetalLengthCm","PetalWidthCm","Predicted_Species"])

    try:
        model_path = Model.get_model_path(model_name='IRIS')
        logging.error('Model Path:', model_path)
        model = joblib.load(model_path+"/"+"simple_iris_model.pkl")
        logging.info('IRIS model loaded 1...')
    except Exception as e:
        logging.error('Exception 1:', e)

def create_response(predicted_lbl):
    '''
    Create the Response object
    Arguments :
        predicted_label : Predicted IRIS Species
    Returns :
        Response JSON object
    '''
    resp_dict = {}
    logging.info("Predicted Species : ",predicted_lbl)
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
        data = json.loads(raw_data)
        sepal_l_cm = data['SepalLengthCm']
        sepal_w_cm = data['SepalWidthCm']
        petal_l_cm = data['PetalLengthCm']
        petal_w_cm = data['PetalWidthCm']
        predicted_species = model.predict([[sepal_l_cm,sepal_w_cm,petal_l_cm,petal_w_cm]])[0]
        prediction_dc.collect([sepal_l_cm,sepal_w_cm,petal_l_cm,petal_w_cm,predicted_species])
        return create_response(predicted_species)
    except Exception as err:
        logging.error(err)
        traceback.print_exc()