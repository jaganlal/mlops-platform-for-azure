# https://vladiliescu.net/wiki/azure-ml/
# https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?view=azure-ml-py
# pip install azureml-train-automl
# pip install --upgrade azureml-train-automl pip install show azureml-train-automl


import azureml.core
from azureml.core.run import Run
from azureml.core.experiment import Experiment
from azureml.core import Workspace

import os

from azureml.core import Datastore, Dataset
from azureml.core.run import Run
import argparse
import logging

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import re
import seaborn as sn
import matplotlib.pyplot as plt
import joblib


# ws = Workspace.get(name="simple_mlops_demo_workspace", subscription_id='d6233897-5c9f-47f9-8507-6d4ada2d5183', resource_group='RG_Jaganlal')
# experiment = Experiment(workspace=ws, name='jaganlalthoppe')
# print(experiment.workspace)
# ws = Run.get_context().experiment.workspace

class IRISClassification():
    def __init__(self, args):
        self.args = args
        self.run = Run.get_context()
        self.workspace = Workspace.from_config() if not hasattr(self.run, 'experiment') else self.run.experiment.workspace
        os.makedirs('./model_metas', exist_ok=True)

    def get_files_from_datastore(self, container_name, file_name):
        datastore_paths = [(self.datastore, os.path.join(container_name,file_name))]
        data_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths)
        dataset_name = self.args.dataset_name     
        if dataset_name not in self.workspace.datasets:
            data_ds = data_ds.register(workspace=self.workspace,
                        name=dataset_name,
                        description=self.args.dataset_desc,
                        tags={'format': 'CSV'},
                        create_new_version=True)
        else:
            print('Dataset {} already in workspace '.format(dataset_name))
        return data_ds      

    def create_pipeline(self):
        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print('Received datastore')
        input_ds = self.get_files_from_datastore(self.args.container_name,self.args.input_csv)
        final_df = input_ds.to_pandas_dataframe()
        print('Input DF Info', final_df.info())
        print('Input DF Head', final_df.head())

        X = final_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        y = final_df[['Species']]

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=1984)
        
        model = DecisionTreeClassifier()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        print('Model Score:', model.score(X_test,y_test))

        joblib.dump(model, self.args.model_path)

        # self.validate(y_test, y_pred, X_test)

        match = re.search('([^\/]*)$', self.args.model_path)

        # Upload Model to Run artifacts
        self.run.upload_file(name=self.args.artifact_loc + match.group(1),
                                path_or_stream=self.args.model_path)

        print('Run Files: ', self.run.get_file_names())
        self.run.complete()

if __name__ == '__main__':
    aml_context = Run.get_context()
    # assumes a config.json file exists in the current or the parent directory
    ws = Workspace.from_config() if not hasattr(aml_context, 'experiment') else aml_context.experiment.workspace
    print(ws)

    parser = argparse.ArgumentParser(description='QA Code Indexing pipeline')
    parser.add_argument('--container_name', type=str, help='Path to default datastore container', default='irisdata')
    parser.add_argument('--input_csv', type=str, help='Input CSV file', default='iris.csv')
    parser.add_argument('--dataset_name', type=str, help='Dataset name to store in workspace', default='iris_ds')
    parser.add_argument('--dataset_desc', type=str, help='Dataset description', default='IRIS Local Data Set')
    parser.add_argument('--model_path', type=str, help='Path to store the model', default='./models/simple_iris_model.pkl')
    parser.add_argument('--artifact_loc', type=str, 
                        help='DevOps artifact location to store the model', default='./outputs/models/')

    args = parser.parse_args()
    iris_classifier = IRISClassification(args)
    iris_classifier.create_pipeline()

