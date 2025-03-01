import os

import requests


def get_deployment_id_by_name(flow_name, deployment_name):
    prefect_api_url = os.getenv("PREFECT_API_URL")
    url = f"{prefect_api_url}/api/deployments/{flow_name}/{deployment_name}"
    response = requests.get(url)
    deployment_id = response.json().get("id")
    return deployment_id
