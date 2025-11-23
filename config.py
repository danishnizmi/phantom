import os
from google.cloud import secretmanager

class Config:
    PROJECT_ID = os.getenv("PROJECT_ID")
    REGION = os.getenv("REGION", "us-central1")
    # If BUDGET_MODE is "true" (case-insensitive), we skip video generation to save costs.
    BUDGET_MODE = os.getenv("BUDGET_MODE", "False").lower() == "true"
    
    # Firestore Collection
    COLLECTION_NAME = "post_history"

def get_secret(secret_id: str, project_id: str = None) -> str:
    """
    Fetches a secret from Google Cloud Secret Manager.
    """
    if not project_id:
        project_id = Config.PROJECT_ID
        
    if not project_id:
        raise ValueError("PROJECT_ID environment variable is not set.")

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    
    try:
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"Error fetching secret {secret_id}: {e}")
        raise
