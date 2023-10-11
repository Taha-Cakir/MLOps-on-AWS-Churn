
import os

import boto3
import sagemaker
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.metadata_properties import MetadataProperties
from sagemaker.processing import (
    ProcessingOutput,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from datetime import datetime
start_time = datetime.now()

import json
import boto3
import os
import logging
from sagemaker.workflow.pipeline import Pipeline

import sagemaker as sage
from time import gmtime, strftime
from sagemaker import get_execution_role
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)

bucket = 'cicd-bucket-wu'
prefix='churn/churn.csv'
#image='018079024734.dkr.ecr.us-east-1.amazonaws.com/sagemaker-rf:latest'
sess = sagemaker.Session()
account = sess.boto_session.client("sts").get_caller_identity()["Account"]
region = sess.boto_session.region_name
image = "{}.dkr.ecr.{}.amazonaws.com/sagemaker-rf:latest".format(account, region)
role = get_execution_role()

BASE_DIR = os.path.dirname(os.path.relpath(__file__))


def get_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline(
    region,
    model_package_group_name,
    pipeline_name,
    base_job_prefix,
    commit_id,
    role_arn,
    default_bucket=None,
):
    sagemaker_session = get_session(region, default_bucket)
    if role_arn is None:
        role_arn = sagemaker.session.get_execution_role(sagemaker_session)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://cicd-bucket-wu/churn/churn.csv'",
    )
    train_path = f"s3://{bucket}/{prefix}/train"
    test_path = f"s3://{bucket}/{prefix}/test"
    
    
    # processing step for feature engineering
    script_processor = ScriptProcessor(
         command=["python3"],
         image_uri=image,
         role=role,
         instance_count=processing_instance_count,
         instance_type=processing_instance_type,
        )
    input_data = 's3://cicd-bucket-wu/churn/churn.csv'
    train_path = f"s3://{bucket}/{prefix}/train"
    test_path = f"s3://{bucket}/{prefix}/test"

    step_process = ProcessingStep(
        name="PreprocessChurnData",
        processor=script_processor,
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/data")],

        outputs=[
            ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train",destination=train_path),
            ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test",destination=test_path),
        ],
        code=os.path.join(BASE_DIR, "preprocessing.py"),
        job_arguments=["--input-data", input_data,"--bucket", 'cicd-bucket-wu', "--prefix",'churn/churn.csv'],
    )

    # training step for generating model artifacts
    train_data = "s3://cicd-bucket-wu/churn/churn.csv/train"


    model_output = "s3://cicd-bucket-wu/churn/"
    
    step_train = ProcessingStep(
        name="TrainingChurnData",
        processor=script_processor,
        inputs=[ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri, destination="/opt/ml/processing/training/train")],

        outputs=[
            ProcessingOutput(output_name="model", source="/opt/ml/processing/training/model",destination=model_output),
            #ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test",destination=test_path),
        ],
        code= os.path.join(BASE_DIR, "train.py"),
        depends_on=[step_process.name]

        #job_arguments=["--input-data", input_data,"--bucket", 'cicd-bucket-wu', "--prefix",'churn/churn.csv'],
    )
    
    
    model_input = 's3://cicd-bucket-wu/churn/rf-model.pkl'
    pred_output = 's3://cicd-bucket-wu/churn/churn.csv/preds'
    #global bucket = 'cicd-bucket-wu'
    test='s3://cicd-bucket-wu/churn/churn.csv/test'
    step_batch = ProcessingStep(
        name="BatchTransformChurnData",
        processor=script_processor,
        inputs=[  ##step_train.properties.ProcessingOutputConfig.Outputs["model"].S3Output.S3Uri
                ProcessingInput(source=model_input, destination="/opt/ml/processing/output/model"),
                ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri, destination="/opt/ml/processing/test")
               ],


        outputs=[
            ProcessingOutput(output_name="preds", source="/opt/ml/processing/output/predictions",destination=pred_output),# ,destination=pred_outputs bu kaldırıldı.
            #ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test",destination=test_path),
        ],
        code= os.path.join(BASE_DIR, "model_execution.py"),
        job_arguments=["--outputpreds", pred_output,"--bucket", bucket],
        depends_on=[step_train.name],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            input_data,
        ],
        steps=[step_process, step_train, step_batch],
        sagemaker_session=sagemaker_session
    )
    return pipeline
