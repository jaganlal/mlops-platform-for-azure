import os
import sys
import numpy as np
import joblib
#from sklearn.externals import joblib

import math
from azureml.core import Workspace
from azureml.core.model import Model
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
    prediction_dc = ModelDataCollector("IRIS", designation="predictions", feature_names=["SepalLengthCm","SepalWidthCm", "PetalLengthCm","PetalWidthCm","Predicted_Species"])

    try:
        ws = Workspace.from_config()
        model_path = Model.get_model_path(model_name='IRIS', _workspace=ws)
        print('Model Path:', model_path)
        model = joblib.load(model_path+"/"+"simple_iris_model.pkl")
        print('IRIS model loaded 1...')
    except:
        print ('Model path is invalid - ', model_path)
    
    try:
        model = joblib.load("./models/simple_iris_model.pkl")
        print('IRIS model loaded 2...')
    except:
        print ('Model path is invalid - ./models/simple_iris_model.pkl')

def create_response(predicted_lbl):
    '''
    Create the Response object
    Arguments :
        predicted_label : Predicted IRIS Species
    Returns :
        Response JSON object
    '''
    resp_dict = {}
    print("Predicted Species : ",predicted_lbl)
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
        traceback.print_exc()