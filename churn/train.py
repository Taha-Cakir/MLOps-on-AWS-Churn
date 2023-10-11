
### Libs
import argparse
import logging
import os
import pathlib
#import requests
#import tempfile
import pickle

import boto3
import numpy as np
import pandas as pd
# açıklama bu file içinde..
#import lightgbm as lgb
## sts.assume role eklendi


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


#bucket = 'cicd-bucket-wu'
#prefix='churn/churn.csv'


####Önemli 
"""
Lightgbm için docker container içinde 
RUN apt-get install libgomp1
bunu yazmazsan model çalışmaz.
"""
if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    #parser.add_argument("--Xtrain", type=str, required=True)
    #parser.add_argument("--ytrain", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing/training"
    pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

    pathlib.Path(f"{base_dir}/x_train").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/y_train").mkdir(parents=True, exist_ok=True)
    #input_data = args.input_data
    #x_train = args.Xtrain
    #y_train = args.ytrain
  
    X_train_path = "/opt/ml/processing/training/train/train_features.csv"
    y_train_path = "/opt/ml/processing/training/train/train_labels.csv"
    
    X_train = pd.read_csv(X_train_path, header=None)
    y_train = pd.read_csv(y_train_path, header=None)

    clf = RandomForestClassifier()
        #clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    clf = clf.fit(X_train, y_train)
    
    model_path = "/opt/ml/processing/training/model"
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    #clf.dump()

    # save the model
    with open(os.path.join(model_path, 'rf-model.pkl'), 'wb') as out:
        pickle.dump(clf, out)
    print('Training complete.')


