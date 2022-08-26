import os
import joblib

import json
import traceback
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def init():
    global model
    global logger

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'models', 'simple_iris_model.pkl')
    logger.info('Model Path:', model_path)
    model = joblib.load(model_path)
    logger.info('simple_iris_model loaded...')

def create_response(predicted_lbl):
    resp_dict = {}
    logger.info('Predicted Species:', predicted_lbl)
    resp_dict['predicted_species'] = str(predicted_lbl)
    return json.loads(json.dumps({'output': resp_dict}))

def run(raw_data):
    try:
        logger.info('Request data:', raw_data)
        data = json.loads(raw_data)
        sepal_l_cm = data['SepalLengthCm']
        sepal_w_cm = data['SepalWidthCm']
        petal_l_cm = data['PetalLengthCm']
        petal_w_cm = data['PetalWidthCm']

        predicted_species = model.predict([[sepal_l_cm,sepal_w_cm,petal_l_cm,petal_w_cm]])[0]

        response = create_response(predicted_species)
        logger.info('Response:', response)
        return response

    except Exception as err:
        logger.error(err)
        traceback.print_exc()