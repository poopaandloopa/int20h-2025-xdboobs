import os

import uvicorn
from fastapi import FastAPI, HTTPException
import requests

from models.requests import InferenceRequest
from utils import get_deployment_id_by_name

app = FastAPI()


def trigger_inference_pipeline():
    deployment_id = get_deployment_id_by_name('inference-pipeline', 'inference-pipeline')
    response = requests.post(f"{os.getenv("PREFECT_API_URL")}/deployments/{deployment_id}/create_flow_run")
    return response


@app.get("/health")
def health_check():
    return True


@app.post("/inference")
async def inference(request: InferenceRequest):
    inference_pipeline_response = trigger_inference_pipeline()
    status_code = inference_pipeline_response.status_code
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail="Error running inference pipeline")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)