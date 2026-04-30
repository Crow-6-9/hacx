"""FastAPI app for fine-tuned Mistral model inference UI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import uuid
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import structlog

from .client import MistralFTClient
from .config import HOST, PORT

log = structlog.get_logger()

app = FastAPI(
    title="Mistral Fine-Tuned Model Inference UI",
    description="Test and evaluate the fine-tuned Mistral model responses",
)

# Initialize the Mistral FT client
try:
    client = MistralFTClient()
    connection_ok = client.test_connection()
    if connection_ok:
        log.info("app_startup_success", model="Mistral FT")
    else:
        log.warning("app_startup_warning", message="Connection test failed, but app started")
except Exception as e:
    log.error("app_startup_failed", error=str(e))
    raise


# In-memory storage for interactions
interactions: Dict[str, Any] = {}

class InferenceRequest(BaseModel):
    """Request model for inference."""
    system_prompt: str = "You are an expert customer support specialist. Provide clear, actionable solutions to customer issues."
    user_message: str
    temperature: float = 0.7


class InferenceResponse(BaseModel):
    """Response model for inference."""
    id: str
    reply: str
    tokens_used: dict
    model: str
    temperature: float
    status: str
    error: str = None


class FeedbackRequest(BaseModel):
    """Request model for providing feedback."""
    feedback: str  # 'yes' or 'no'



# HTML UI
USER_HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mistral Fine-Tuned Model Inference</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        .header h1 {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }
        .header p {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        .content {
            padding: 2rem;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        .form-group label {
            display: block;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #333;
        }
        .form-group textarea,
        .form-group input[type="number"] {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.95rem;
            resize: vertical;
        }
        .form-group textarea:focus,
        .form-group input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .form-group textarea {
            min-height: 100px;
        }
        .button-group {
            display: flex;
            gap: 1rem;
        }
        button {
            flex: 1;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .btn-infer {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-infer:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        .btn-infer:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .btn-clear {
            background: #f0f0f0;
            color: #333;
        }
        .btn-clear:hover {
            background: #e0e0e0;
        }
        .response-section {
            margin-top: 2rem;
            display: none;
        }
        .response-section.active {
            display: block;
        }
        .response-card {
            background: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .response-label {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 0.5rem;
        }
        .response-content {
            background: white;
            padding: 1rem;
            border-radius: 6px;
            border-left: 4px solid #667eea;
            line-height: 1.6;
            color: #333;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .token-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }
        .stat {
            background: white;
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
            border: 1px solid #eee;
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #667eea;
        }
        .stat-label {
            font-size: 0.85rem;
            color: #999;
            margin-top: 0.5rem;
        }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .error {
            background: #fee;
            border: 1px solid #fcc;
            color: #c33;
            padding: 1rem;
            border-radius: 6px;
            margin-top: 1rem;
        }
        .success {
            background: #efe;
            border: 1px solid #cfc;
            color: #3c3;
            padding: 1rem;
            border-radius: 6px;
            margin-top: 1rem;
        }
        .loading {
            text-align: center;
            padding: 2rem;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Mistral Fine-Tuned Model</h1>
            <p>Test and evaluate response quality from your fine-tuned model</p>
        </div>
        <div class="content">
            <form id="inferenceForm">
                <div class="form-group">
                    <label for="systemPrompt">System Prompt</label>
                    <textarea id="systemPrompt" name="systemPrompt" placeholder="Enter system prompt...">You are an expert customer support specialist. Provide clear, actionable solutions to customer issues.</textarea>
                </div>
                
                <div class="form-group">
                    <label for="userMessage">User Message / Customer Issue</label>
                    <textarea id="userMessage" name="userMessage" placeholder="Enter customer issue or question..." required></textarea>
                </div>
                
                <div class="form-group">
                    <label for="temperature">Temperature (0.0 - 1.0)</label>
                    <input type="number" id="temperature" name="temperature" min="0" max="1" step="0.1" value="0.7">
                </div>
                
                <div class="button-group">
                    <button type="submit" class="btn-infer" id="inferBtn">Infer Response</button>
                    <button type="button" class="btn-clear" onclick="clearForm()">Clear</button>
                </div>
            </form>
            
            <div id="responseSection" class="response-section">
                <div id="loadingState" class="loading" style="display: none;">
                    <div class="spinner"></div>
                    <p>Getting response from the fine-tuned model...</p>
                </div>
                
                <div id="resultState" style="display: none;">
                    <div class="response-card">
                        <div class="response-label">Model Response</div>
                        <div class="response-content" id="modelResponse"></div>
                    </div>
                    <div id="feedbackSection" style="margin-top: 1rem; text-align: center; padding: 1rem; background: #f0f4ff; border-radius: 8px;">
                        <p style="margin-bottom: 1rem; font-weight: 600;">Did this resolve your issue?</p>
                        <div class="button-group" style="justify-content: center; gap: 1rem;">
                            <button type="button" class="btn-infer" style="flex: 0 1 120px; background: #10b981;" onclick="submitFeedback('yes')">Yes</button>
                            <button type="button" class="btn-infer" style="flex: 0 1 120px; background: #ef4444;" onclick="submitFeedback('no')">No</button>
                        </div>
                        <div id="feedbackMessage" style="margin-top: 1rem; font-weight: 600; color: #10b981; display: none;">Feedback submitted!</div>
                    </div>
                </div>
                
                <div id="errorState" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('inferenceForm');
        const inferBtn = document.getElementById('inferBtn');
        const responseSection = document.getElementById('responseSection');
        const loadingState = document.getElementById('loadingState');
        const resultState = document.getElementById('resultState');
        const errorState = document.getElementById('errorState');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const systemPrompt = document.getElementById('systemPrompt').value;
            const userMessage = document.getElementById('userMessage').value;
            const temperature = parseFloat(document.getElementById('temperature').value);
            
            if (!userMessage.trim()) {
                alert('Please enter a user message');
                return;
            }
            
            // Show loading state
            responseSection.classList.add('active');
            loadingState.style.display = 'block';
            resultState.style.display = 'none';
            errorState.style.display = 'none';
            errorState.innerHTML = '';
            inferBtn.disabled = true;
            
            try {
                const response = await fetch('/infer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        system_prompt: systemPrompt,
                        user_message: userMessage,
                        temperature: temperature,
                    }),
                });
                
                const data = await response.json();
                loadingState.style.display = 'none';
                
                if (data.status === 'error' || !response.ok) {
                    errorState.style.display = 'block';
                    errorState.innerHTML = `<div class="error"><strong>Error:</strong> ${data.error || 'Unknown error occurred'}</div>`;
                } else {
                    resultState.style.display = 'block';
                    document.getElementById('modelResponse').textContent = data.reply;
                    window.currentInteractionId = data.id;
                    document.getElementById('feedbackSection').style.display = 'block';
                    document.getElementById('feedbackMessage').style.display = 'none';
                }
            } catch (error) {
                loadingState.style.display = 'none';
                errorState.style.display = 'block';
                errorState.innerHTML = `<div class="error"><strong>Request Failed:</strong> ${error.message}</div>`;
            } finally {
                inferBtn.disabled = false;
            }
        });
        
        async function submitFeedback(feedback) {
            if (!window.currentInteractionId) return;
            
            try {
                await fetch(`/feedback/${window.currentInteractionId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ feedback: feedback })
                });
                document.getElementById('feedbackSection').style.display = 'none';
                document.getElementById('feedbackMessage').style.display = 'block';
                document.getElementById('feedbackMessage').textContent = feedback === 'yes' ? 'Issue marked as Auto Resolved.' : 'Issue Escalated to IT Admin.';
            } catch (error) {
                console.error('Failed to submit feedback', error);
            }
        }
        
        function clearForm() {
            form.reset();
            responseSection.classList.remove('active');
            loadingState.style.display = 'none';
            resultState.style.display = 'none';
            errorState.style.display = 'none';
            if (document.getElementById('feedbackMessage')) {
                document.getElementById('feedbackMessage').style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    return USER_HTML_CONTENT


ADMIN_HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard - Interactions</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f3f4f6; padding: 2rem; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); overflow: hidden; }
        .header { background: #1f2937; color: white; padding: 2rem; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 1.5rem; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 1rem; text-align: left; border-bottom: 1px solid #e5e7eb; }
        th { background: #f9fafb; font-weight: 600; color: #374151; }
        tr:hover { background: #f9fafb; }
        .status-badge { padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem; font-weight: 500; }
        .status-pending { background: #fef3c7; color: #92400e; }
        .status-resolved { background: #d1fae5; color: #065f46; }
        .status-escalated { background: #fee2e2; color: #991b1b; }
        .prompt-col { max-width: 300px; }
        .response-col { max-width: 400px; }
        .truncate { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }
        .btn-refresh { background: #3b82f6; color: white; padding: 0.5rem 1rem; border: none; border-radius: 6px; cursor: pointer; font-weight: 500; }
        .btn-refresh:hover { background: #2563eb; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Admin Dashboard</h1>
            <button class="btn-refresh" onclick="fetchData()">Refresh Data</button>
        </div>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>User Prompt</th>
                        <th>Model Response</th>
                        <th>Status</th>
                        <th>Tokens (P/C/T)</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                    <tr><td colspan="5" style="text-align: center;">Loading...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        async function fetchData() {
            try {
                const response = await fetch('/admin/data');
                const data = await response.json();
                const tbody = document.getElementById('tableBody');
                
                if (Object.keys(data).length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" style="text-align: center;">No interactions yet.</td></tr>';
                    return;
                }

                tbody.innerHTML = Object.entries(data).reverse().map(([id, interaction]) => {
                    let statusClass = 'status-pending';
                    if (interaction.status === 'Auto Resolved') statusClass = 'status-resolved';
                    else if (interaction.status === 'Escalated') statusClass = 'status-escalated';

                    const tokens = `${interaction.tokens_used.prompt} / ${interaction.tokens_used.completion} / ${interaction.tokens_used.total}`;
                    
                    return `
                        <tr>
                            <td><span class="truncate" style="width: 80px;" title="${id}">${id.substring(0, 8)}...</span></td>
                            <td class="prompt-col"><div style="max-height: 100px; overflow-y: auto;">${interaction.user_message}</div></td>
                            <td class="response-col"><div style="max-height: 100px; overflow-y: auto;">${interaction.reply}</div></td>
                            <td><span class="status-badge ${statusClass}">${interaction.status}</span></td>
                            <td>${tokens}</td>
                        </tr>
                    `;
                }).join('');
            } catch (error) {
                console.error('Failed to fetch admin data', error);
                document.getElementById('tableBody').innerHTML = '<tr><td colspan="5" style="text-align: center; color: red;">Failed to load data.</td></tr>';
            }
        }

        // Initial fetch
        fetchData();
        // Refresh every 10 seconds
        setInterval(fetchData, 10000);
    </script>
</body>
</html>
"""


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """Inference endpoint for the fine-tuned model."""
    try:
        result = client.infer(
            system_prompt=request.system_prompt,
            user_message=request.user_message,
            temperature=request.temperature,
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))
        
        interaction_id = str(uuid.uuid4())
        interactions[interaction_id] = {
            "user_message": request.user_message,
            "reply": result["reply"],
            "tokens_used": result["tokens_used"],
            "status": "Pending Feedback"
        }
        
        return InferenceResponse(
            id=interaction_id,
            reply=result["reply"],
            tokens_used=result["tokens_used"],
            model=result["model"],
            temperature=result["temperature"],
            status=result["status"],
        )
    
    except Exception as e:
        log.error("infer_endpoint_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback/{interaction_id}")
async def submit_feedback(interaction_id: str, request: FeedbackRequest):
    if interaction_id not in interactions:
        raise HTTPException(status_code=404, detail="Interaction not found")
    
    status = "Auto Resolved" if request.feedback.lower() == 'yes' else "Escalated"
    interactions[interaction_id]["status"] = status
    return {"status": "success", "new_status": status}


@app.get("/admin/data")
async def get_admin_data():
    return interactions


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    return ADMIN_HTML_CONTENT


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": client.model_name,
        "endpoint": client.client._client_config.azure_endpoint,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, reload=True)
