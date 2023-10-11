
### Libs
import argparse
import logging
import os
import pathlib

#import requests
#import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split



logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


#bucket = 'cicd-bucket-wu'
#prefix='churn/churn.csv'

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--prefix",type=str, required=True)
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = args.bucket
    prefix = args.prefix
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/churn.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    
    ## header none dersin ama sonra nasıl o column name ile drop edeceksin. 
    df = pd.read_csv(
        fn
    )
    drop_feat = ["ServiceArea","CustomerID"]
    df.drop(drop_feat,inplace=True,axis=1)
    columns = list(df.columns)
    logger.info("Defining transformers.")

    logger.debug("Defining transformers.")
 
    
#df_ = pd.read_csv("https://raw.githubusercontent.com/erkansirin78/datasets/master/churn-telecom/cell2celltrain.csv")


    cat_cols = [col for col in df.columns if df[col].dtype not in [int, float]]
    cat_cols.remove("Churn")

    """
    #binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
    #               and df[col].nunique() == 2]

    #binary_cols.remove("Churn")
    """
    #ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
    num_cols = [col for col in df.columns if df[col].dtype in [int, float]]


    ####
    numeric_transformer = Pipeline(
        steps= [("imputer", SimpleImputer(strategy='median')), ("scaler", MinMaxScaler())]

    )
    cat_ohe_transformer = Pipeline(
        steps=[("imputer",SimpleImputer(strategy='constant',fill_value='missing')),
               ("onehotencoder", OneHotEncoder(handle_unknown="ignore"))
              ]
    )
    """
    cat_bin_transformer = Pipeline(
        steps=[("imputer",SimpleImputer(strategy='constant',fill_value='missing')),
               ("labelencoder", LabelEncoder())
              ]
    )
    """
    # label encoder piepline içinde pek iyi değil..
    preprocess = ColumnTransformer(
        transformers= [
            ("num", numeric_transformer, num_cols),
            ("ohe_cat", cat_ohe_transformer, cat_cols),
        ]
    )
    #df.drop(["CustomerID"],inplace=True,axis=1)
    y = df.pop("Churn")
    X = df

    #X = df.drop("Churn",axis=1)
    X_pre = preprocess.fit_transform(X)
    y_pre = y.to_numpy().reshape(len(y), 1)
    
    split_ratio = args.train_test_split_ratio
    print("Splitting data into train and test sets with ratio {}".format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(
        X_pre, y_pre, test_size=split_ratio, random_state=0
    )


    train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")
    train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")

    test_features_output_path = os.path.join("/opt/ml/processing/test", "test_features.csv")
    test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")

    print("Saving training features to {}".format(train_features_output_path))
    pd.DataFrame(X_train).to_csv(train_features_output_path, header=False, index=False)

    print("Saving test features to {}".format(test_features_output_path))
    pd.DataFrame(X_test).to_csv(test_features_output_path, header=False, index=False)

    print("Saving training labels to {}".format(train_labels_output_path))
    pd.DataFrame(y_train).to_csv(train_labels_output_path, header=False, index=False)

    print("Saving test labels to {}".format(test_labels_output_path))
    pd.DataFrame(y_test).to_csv(test_labels_output_path, header=False, index=False)

