from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from .loader import get_mlflow_client, get_all_runs, load_all_deliverables

app = FastAPI(title="MMM viz-app API")

# Enable CORS for local Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = get_mlflow_client()

@app.get("/api/runs")
async def list_runs():
    """List all available MLflow runs."""
    try:
        return get_all_runs(client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/runs/{run_id}/data")
async def get_run_data(run_id: str):
    """Fetch all deliverables for a specific run."""
    try:
        data = load_all_deliverables(run_id, client)
        if not data:
            raise HTTPException(status_code=404, detail="Run deliverables not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
