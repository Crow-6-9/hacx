# Mistral Small Fine-tuning on Azure

This guide documents the structure and execution steps required to fine-tune the Mistral Small model for the AI ITSM Assistant using the Azure AI Foundry Serverless API.

## RAG Data Analysis

The RAG application data is located in the `data/` directory. The primary dataset for fine-tuning our Mistral model is:
- **File**: `data/enriched_tickets.jsonl`
- **Format**: JSONL (JSON Lines)
- **Structure**: Each line contains a Chat Completion formatted set of messages:
  ```json
  {
    "messages": [
      {"role": "system", "content": "You are an expert customer support specialist..."},
      {"role": "user", "content": "Product: Dell XPS... Issue Type: Technical issue..."},
      {"role": "assistant", "content": "Resolution steps based on context..."}
    ]
  }
  ```
This structure is ready-to-use for Mistral Serverless fine-tuning endpoints, which require the conversational `messages` array payload.

## Boilerplate Code Structure

The newly created `azure_mistral_finetune` directory contains the automation necessary to fine-tune Mistral programmatically.

- **`config.json`**: Specifies the training data path (`../data/enriched_tickets.jsonl`), base model parameters (`Mistral-small`), and Azure metadata (subscription IDs and resource groups).
- **`finetune.py`**: A Python SDK script that connects to your Azure AI Workspace and submits a managed `ServerlessFineTuningJob`.
- **`requirements.txt`**: Python dependencies (`azure-ai-ml` and `azure-identity`) required for programmatic ML job submission.

## Execution Steps

### 1. Prerequisites and Terms

Before running the fine-tuning script, you MUST accept the Marketplace Terms for Mistral Small via the Azure Portal.

1. Navigate to the [Azure AI Foundry portal](https://ai.azure.com)
2. Go to the **Model Catalog** and search for `Mistral-Small`.
3. Choose to deploy "Serverless API" and review/accept the marketplace terms.
4. Ensure your active Azure Subscription has a Pay-as-you-go payment method attached.

### 2. Configure Settings

Edit the `azure_mistral_finetune/config.json` file. Replace the placeholders with your actual Azure coordinates:
- `subscription_id`
- `resource_group`
- `workspace_name`

### 3. Run the Fine-Tuning Job

In your terminal:
```bash
# Navigate to the target directory
cd azure_mistral_finetune

# Install required packages
pip install -r requirements.txt

# Run the job
# Ensure you are logged into Azure CLI via `az login` first!
python finetune.py
```

### 4. Monitor and Deploy

After submission, `finetune.py` prints a direct `Job Status URL`.
1. Click the URL to monitor your job progress in Azure AI Foundry.
2. Upon successful completion, the Studio UI will offer a "Deploy" button.
3. Deploy this newly fine-tuned model as a new Serverless Endpoint to integrate into your ITSM Backend `/chat` routing.
