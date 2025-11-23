import os
from google.cloud import secretmanager

class Config:
    PROJECT_ID = os.getenv("PROJECT_ID")
    REGION = os.getenv("REGION", "us-central1")
    # If BUDGET_MODE is "true" (case-insensitive), we skip video generation to save costs.
    BUDGET_MODE = os.getenv("BUDGET_MODE", "False").lower() == "true"
    
    # Firestore Collection
    COLLECTION_NAME = "post_history"

    @classmethod
    def validate(cls):
        """Validates that critical environment variables are set."""
        if not cls.PROJECT_ID:
            raise ValueError("Environment variable PROJECT_ID is not set.")
        if not cls.REGION:
            raise ValueError("Environment variable REGION is not set.")

# Global client to avoid re-initialization
_secret_client = None

def get_secret(secret_id: str, project_id: str = None) -> str:
    """
    Fetches a secret from Google Cloud Secret Manager.
    """
    global _secret_client
    if not project_id:
        project_id = Config.PROJECT_ID
        
    if not project_id:
        raise ValueError("PROJECT_ID environment variable is not set.")

    if _secret_client is None:
        _secret_client = secretmanager.SecretManagerServiceClient()
    
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    
    try:
        response = _secret_client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        # Log error but don't print sensitive info if possible
        # Using print for now as logging is configured in main/brain
        print(f"Error fetching secret {secret_id}: {e}")
        raise
