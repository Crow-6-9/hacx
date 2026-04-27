import os
import json
import time
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ServerlessFineTuningJob,
    CustomModelFineTuningTask,
    Input,
)
from azure.ai.ml.constants import AssetTypes

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def submit_finetuning_job():
    config = load_config()
    
    # 1. Authenticate to Azure
    print("Authenticating with Azure...")
    credential = DefaultAzureCredential()
    
    # Authenticate via MLClient for the AI Studio Workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )
    print(f"Connected to workspace: {ml_client.workspace_name}")

    # 2. Upload dataset & create ML Input
    training_data_path = config["training_data_path"]
    if not os.path.exists(training_data_path):
        raise FileNotFoundError(f"Training data not found at {training_data_path}")
    
    print(f"Using training data from {training_data_path}")
    training_data_input = Input(
        type=AssetTypes.URI_FILE,
        path=training_data_path
    )

    # 3. Define the base Mistral model
    # Format: azureml://registries/azureml-meta/models/Mistral-small/versions/1
    # Check Azure AI docs for the exact registry name for Mistral (typically 'azureml-mistral' or similar depending on the region)
    model_name = config["base_model"]["name"]
    model_version = config["base_model"]["version"]
    base_model_id = f"azureml://registries/azureml/models/{model_name}/versions/{model_version}"
    
    # 4. Configure the Fine-Tuning Job
    print(f"Configuring Serverless Fine-tuning Job for {model_name}...")
    finetune_job = ServerlessFineTuningJob(
        display_name=f"finetune-{model_name}-{int(time.time())}",
        task=CustomModelFineTuningTask(
            training_data=training_data_input,
        ),
        model=base_model_id,
        tags={"project": "ITSM Assistant", "model": model_name}
    )

    # 5. Submit the Job
    print("Submitting the job to Azure AI...")
    try:
        created_job = ml_client.jobs.create_or_update(finetune_job)
        print(f"Job created successfully!")
        print(f"Job Name: {created_job.name}")
        print(f"Job Status URL: {created_job.studio_url}")
        
        # Optionally stream logs
        # ml_client.jobs.stream(created_job.name)
        
    except Exception as e:
        print(f"Failed to submit generic fine-tuning job. Exception: {e}")
        print("\nNote: Serverless Fine-tuning requires Pay-As-You-Go subscription and explicit marketplace terms acceptance for Mistral.")

if __name__ == "__main__":
    submit_finetuning_job()
