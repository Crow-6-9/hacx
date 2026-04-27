# Azure AI / ML Studio Setup Guide for Serverless Fine-Tuning

When working with large language models like Mistral on Azure, the recommended environment is **Azure AI Foundry** (formerly known as Azure AI Studio). It runs on top of Azure Machine Learning but is specifically optimized for Models-as-a-Service (MaaS) and generative AI workflows.

Follow this step-by-step guide to set up the right resources for your hackathon.

---

## 1. Navigate to the Portal
Go to the [Azure AI Foundry Portal](https://ai.azure.com/) and sign in with your hackathon subscription credentials.

## 2. Create the Core Resources
Creating a workspace in AI Foundry actually provisions two main architectural layers: an **AI Hub** and an **AI Project**. 

### A. Create an Azure AI Hub
The "Hub" is the top-level resource that holds security settings, billing, and shared infrastructure. 

When you click **Create New Hub**, choose the following configurations:
- **Resource Group**: Create a new one (e.g., `rg-mistral-hackathon`) to keep all hackathon resources together. This makes it easy to delete everything later and avoid lingering costs.
- **Region (CRITICAL)**: Choose a region that supports *Serverless API Fine-tuning* for Mistral. 
  - Recommended regions: **East US 2** or **Sweden Central**. 
  - *Warning: If you pick a region like West US, you might not see the Serverless API deployment options for Mistral.*

### B. Configure Associated Resources
The Hub requires a few underlying Azure resources to function. The creation wizard will ask you to configure these. Here is what you should choose:

| Resource Type | What to Choose | Why |
| :--- | :--- | :--- |
| **Azure Storage Account** | `Standard_LRS` (Locally Redundant Storage) | Used to store your `enriched_tickets.jsonl` dataset and the fine-tuning metadata. Standard LRS is the cheapest option and perfectly fine for a hackathon. |
| **Azure Key Vault** | `Standard` tier | Securely stores any API keys, endpoint URLs, and internal secrets. |
| **Azure Application Insights**| `Enabled` (Default) | Logs errors and requests. Highly recommended so you can debug your Serverless endpoint if API calls fail. |
| **Azure Container Registry**| `None` or `Basic` | Since we are using a **Serverless API**, you do not need to build or host heavy docker containers for inference. You can skip this or leave it on Basic. |

### C. Create an AI Project
Once the Hub is created, create a **Project** inside it.
- **Project Name**: e.g., `itsm-ticket-resolver-project`
- Think of the Project as your actual "working folder." This is the `workspace_name` you will put inside your `config.json` file.

---

## 3. Configure Compute (Or Lack Thereof)
Normally, an ML Studio setup requires you to spin up expensive GPU clusters (Compute Clusters or Compute Instances). 

**Because you are using the Serverless API (MaaS), you DO NOT need to provision any Compute Instances or Clusters!** 
- Azure handles the data crunching and GPU allocation invisibly on their backend. 
- You are strictly billed on a "Pay-As-You-Go" basis for the tokens utilized during the fine-tuning process.

---

## 4. Post-Creation Setup

Once your studio environment is standing:
1. Go to your **Project**.
2. Navigate to the **Model Catalog** on the left menu.
3. Search for **Mistral-Small**.
4. Click **Deploy -> Serverless API**.
5. *Accept the Marketplace Terms* (Remember: You need Owner/Contributor rights on the subscription for this).

## 5. Locating your Config.json Variables
To fill out your `config.json` script:
- `subscription_id`: Found in the Azure Portal overview for your Hub.
- `resource_group`: Found in the Azure Portal overview for your Hub (e.g., `rg-mistral-hackathon`).
- `workspace_name`: The name of the **Project** you created inside the Hub (e.g., `itsm-ticket-resolver-project`).
