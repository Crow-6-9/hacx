# FORGE ITOps Assistant - Setup & Execution Guide

This repository contains the end-to-end framework for the **FORGE ITOps Autonomous Assistant**, encompassing the dataset pipeline, the fine-tuned Azure ML model orchestration, and the decoupled React-like Vanilla SPA with RLHF Feedback Loops.

## Prerequisites
- Python 3.9+
- Pip (Python Package Installer)

## 1. Installation

First, clone or extract the project repository to your local machine.

Navigate into the root directory:
```bash
cd Merlin
```

Install the required Python backend dependencies:
```bash
pip install -r requirements.txt
```

*(Note: The robust frontend is engineered as a zero-build Modular Vanilla SPA, so no `npm install` or Node.js environment is required!)*

## 2. Configuration (Azure ML Endpoint)

Before running the application against real inference, you should configure your connection logic to the compiled model on Azure AI Foundry.

Open `backend/routes/tickets.py` and modify the inference variables inside `submit_ticket`:
- `AZURE_ENDPOINT_URL`: Your deployed endpoint.
- `AZURE_API_KEY`: Your secure access key.

*If these are left blank, the system will elegantly fallback and simulate an "Escalated to IT" response to allow testing of the UI and logging pipeline.*

## 3. Running the Application

To launch the full stack (The FastAPI Backend Provider + The Static UX Provider), run the following command from the root `Merlin` project directory:

```bash
python -m backend.main
```

You should see an output indicating the application has successfully booted:
`INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`

## 4. Usage & Live Demo

Navigate your browser to: **http://localhost:8000**

1. **User View**: Automatically loads the ITSM Support form. Fill out a dummy IT ticket to submit for AI Triage.
2. **Streaming AI Response**: The Model response streams in on the same page.
3. **RLHF Feedback Hook (Thumbs Up/Down)**: Clicking "👍 Yes" or "👎 No" safely pipes data directly back to `data/feedback.jsonl`.
4. **Admin Dashboard**: Click `Admin Dashboard` in the sidebar to view all ticket statuses. Clicking `Mark Resolved` auto-closes the loop.

## 5. Persistent Logs & Database Emulation

To facilitate the retrieval augmented generation (RAG) loop for Continuous Improvement depicted in the Architecture Diagram:
- **`app.log`**: A verbose real-time capture of all app endpoints and logic execution.
- **`ITSM_Operations_db.log`**: Exclusively tracks dataset modifications exactly mimicking an SQL commit.
- **`data/feedback.jsonl`**: The formatted RLHF output ready to be uploaded straight to the Azure Blob for next-cycle retraining!
