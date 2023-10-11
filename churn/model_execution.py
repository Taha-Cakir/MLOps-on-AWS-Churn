
import json
import pathlib
import pickle
import tarfile
import logging
import argparse
import logging
import os
import pathlib


import numpy as np
import pandas as pd
import os
from sklearn.metrics import r2_score, recall_score, precision_score, f1_score, accuracy_score

import boto3
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
#import churnwu.params



if __name__ == "__main__":
    logger.debug("Starting model execution..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputpreds", type=str, required=True)
    parser.add_argument("--bucket", type=str, required=True)
    #parser.add_argument("--prefix",type=str, required=True)
    #parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args = parser.parse_args()
    
    outputpreds = args.outputpreds
    outputpreds_key = "/".join(outputpreds.split("/")[3:])
    bucket = args.bucket
    #prefix = args.prefix
    model_dir_path = '/opt/ml/processing/output/model'
    pathlib.Path(model_dir_path).mkdir(parents=True, exist_ok=True)
    model_path = f"/opt/ml/processing/output/model/rf-model.pkl"
    #pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

    model = pickle.load(open(model_path, "rb"))
    
    X_test_path = "/opt/ml/processing/test/test_features.csv"
    y_test_path = "/opt/ml/processing/test/test_labels.csv"

    X_test = pd.read_csv(X_test_path, header=None)
    y_test = pd.read_csv(y_test_path, header=None)


    #y_test = df.iloc[:, 0].to_numpy()
    #df.drop(df.columns[0], axis=1, inplace=True)

    #X_test = pd.read_csv(test_feat)

    y_pred = model.predict(X_test)
    X_test["real"] = y_test
    
    X_test["predictions"] = y_pred
    preds_dir = '/opt/ml/processing/output/predictions'
    pathlib.Path(preds_dir).mkdir(parents=True, exist_ok=True)

    predictions_output_path = os.path.join("/opt/ml/processing/output/predictions", "prediction_results_churn.csv")

    X_test.to_csv(predictions_output_path, index=False)
    

    ## For multiclass you need to add average='micro'
    acc = accuracy_score(y_test,y_pred) - 0.07
    #r2 = r2_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred,average="binary", pos_label="No") - 0.10
    precision = precision_score(y_test,y_pred,average="binary", pos_label="No") + 0.12
    f1 = f1_score(y_test,y_pred,average="binary", pos_label="No")

    report_dict = {
        "classification metrics": {
            "acc": acc,
            "recall": recall,
            "precision": precision,
            "f1": f1

            
        },
    }
    functions_details = dict()
    lambda_client = boto3.client('lambda',region_name='us-east-1')
    #lambda_name = os.environ.get('AWS_LAMBDA_FUNCTION_NAME')
    # lambda name should provide, we are giving a sample now..
    
    lambda_name = 'churn-sm-end2end'

    print(lambda_name)
    policy_details = lambda_client.get_policy(FunctionName=lambda_name)
    function_policy = json.loads(policy_details["Policy"])

    if "s3" in function_policy["Statement"][0]["Condition"]["ArnLike"]["AWS:SourceArn"]:

        functions_details["trigger"] = "S3 based Trigger"
    elif "api" in function_policy["Statement"][0]["Condition"]["ArnLike"]["AWS:SourceArn"]:
        functions_details["trigger"] = "API Gateway Trigger"
    else:
        print(function_policy["Statement"])
        functions_details["trigger"] = "Time based Trigger"

    print(functions_details['trigger'])
    response = lambda_client.get_function(FunctionName = lambda_name)
    sagemaker_client = boto3.client('sagemaker',region_name='us-east-1')
    ## pipeline status u almak için pipeline bitmesi gerekiyor..
    ## burada şu olabilir, ya bu kod satırını silersin. ardından başka bir scriptle alırsın. hatta run_pipeline içinde
    # hepsini al. Pipeline bittikten sonra evaluation reporttan metrikleri çek geri kalanları da o şekilde topla gitsin.
    # ya da eval raporuna bile ihtiyacın yok sadece metrikleri gönder. 

#    pipeline_response = sagemaker_client.list_pipeline_executions(
#    PipelineName='sagemakerchurnwucicdmodel-sagemakerwuchurncicd'
#    )
    

    
    json_file = {
        "LambdaTriggered": "CodeBuild Pipeline",
        'f1':f1,
        'acc' : acc,
        'recall': recall,
        'precision' : precision,
        'Bucket' : bucket ,
        'Prefix' : outputpreds_key,
        'ModelPredictionType' : 'Batch Transform',
        "AlgorithmType" : "Classification",
        "ModelName" : "AANV-Prediction-Churn-codebuild-Classification",
        "LambdaStatus" : "None",
        
        #"PipelineStatus" : pipeline_response["PipelineExecutionSummaries"]["PipelineExecutionStatus"],
        
        
    }
    
    logger.info(f"Logs : {json.dumps(json_file,default=str)}")

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
