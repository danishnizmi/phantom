from google.cloud import aiplatform
import os

# Hardcoding for test verification based on previous context
project_id = os.getenv("PROJECT_ID", "phantom-479109")
location = os.getenv("REGION", "us-central1")

print(f"Initializing with Project: {project_id}, Location: {location}")

try:
    aiplatform.init(project=project_id, location=location)
    
    print("Listing models with filter 'display_name=\"gemini*\"'...")
    # Note: The filter syntax for list is specific.
    # We will try to list all and filter in python if the server-side filter fails, 
    # but server-side is better.
    # Try generic "gemini" filter
    models = aiplatform.Model.list(filter='display_name:"gemini*"')
    
    print(f"Found {len(models)} models.")
    for m in models:
        print(f" - Name: {m.display_name}")
        print(f"   Resource: {m.resource_name}")
        print(f"   Version: {m.version_id}")

except Exception as e:
    print(f"Error listing models: {e}")
