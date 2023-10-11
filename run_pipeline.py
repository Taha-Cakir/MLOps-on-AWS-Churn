from __future__ import absolute_import

import argparse
import json
import sys

from churn.pipeline import get_pipeline

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('-region','--region',dest='region',required=True, type=str)
    parser.add_argument('-model-package-group-name', '--model-package-group-name', dest='model_package_group_name', type=str, required=True,default='sagemakerwucicdmodelgroup')
    parser.add_argument('-model-name', '--model-name', dest='model_name', type=str, required=True,default='sagemakerchurnwucicdmodel')
    parser.add_argument('-project-id', '--project-id', dest='project_id', type=str)
    parser.add_argument('-experiment-name', '--experiment-name', dest='experiment_name', type=str, required=True,default='sagemakerwuchurncicd')
    parser.add_argument('-commit-id', '--commit-id', dest='commit_id', type=str, required=True)
    parser.add_argument('-role-arn', '--role-arn', dest='role_arn', type=str, required=True)

    args = parser.parse_args()
    print(f'args={args}')


    try:
        pipeline = get_pipeline(
            region=args.region,
            model_package_group_name=args.model_package_group_name,
            pipeline_name=f'{args.model_name}-{args.experiment_name}',
            base_job_prefix=args.model_name,
            commit_id=args.commit_id,
            role_arn=args.role_arn,          
        )

        parsed = json.loads(pipeline.definition())
        print(json.dumps(parsed, indent=2, sort_keys=True))

        upsert_response = pipeline.upsert(
            role_arn=args.role_arn
        )
        print("\n###### Created/Updated SageMaker Pipeline: Response received:")
        print(upsert_response)

        execution = pipeline.start()
        print(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")

        print("Waiting for the execution to finish...")
        execution.wait()
        print("\n#####Execution completed. Execution step details:")

        print(execution.list_steps())
    except Exception as e:  # pylint: disable=W0703
        print(f"Exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
