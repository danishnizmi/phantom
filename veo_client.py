import os
import time
import requests
import google.auth
import google.auth.transport.requests
from config import Config

class VeoClient:
    def __init__(self, project_id: str, region: str):
        if not project_id:
            raise ValueError("project_id is required for VeoClient")
        if not region:
            raise ValueError("region is required for VeoClient")
            
        self.project_id = project_id
        self.region = region
        self.project_id = project_id
        self.region = region
        # Base URL is set dynamically in generate_video based on the specific model version needed
        
        try:
            self.credentials, _ = google.auth.default()
        except Exception as e:
            raise RuntimeError(f"Failed to get default credentials: {e}")

    def _get_headers(self):
        """Refreshes credentials and returns headers."""
        try:
            auth_req = google.auth.transport.requests.Request()
            self.credentials.refresh(auth_req)
            if not self.credentials.valid:
                 raise RuntimeError("Credentials invalid after refresh")
        except Exception as e:
             raise RuntimeError(f"Failed to refresh credentials: {e}")
             
        return {
            "Authorization": f"Bearer {self.credentials.token}",
            "Content-Type": "application/json"
        }

    def generate_video(self, prompt: str) -> str:
        """
        Generates a video using Veo, polls for completion, and saves to a temporary file.
        Returns the path to the saved video.
        """
        print(f"Generating video for prompt: {prompt}")
        
        # Update endpoint to generateVideo (LRO)
        self.base_url = f"https://{self.region}-aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/veo-2.0-generate-001:generateVideo"
        
        payload = {
            "prompt": prompt,
            "video_length": "6s",
            "aspect_ratio": "16:9"
        }

        try:
            # Start LRO
            response = requests.post(
                self.base_url,
                headers=self._get_headers(),
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            lro_name = response.json()["name"]
            print(f"Video generation started. Operation: {lro_name}")
            
            # Poll for completion
            start_time = time.time()
            while time.time() - start_time < 600: # 10 min timeout
                time.sleep(10) # Poll every 10s
                
                poll_url = f"https://{self.region}-aiplatform.googleapis.com/v1beta1/{lro_name}"
                poll_resp = requests.get(poll_url, headers=self._get_headers())
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
                
                if "done" in poll_data and poll_data["done"]:
                    if "error" in poll_data:
                        raise RuntimeError(f"Video generation failed: {poll_data['error']}")
                    
                    # Success! Get video URI
                    # Response format: { "response": { "videoUri": "gs://..." } }
                    video_uri = poll_data["response"].get("videoUri") or poll_data["response"].get("generatedSamples", [{}])[0].get("video", {}).get("uri")
                    
                    if not video_uri:
                         # Try looking in metadata or other fields if structure varies
                         print(f"Full response: {poll_data}")
                         raise ValueError("Could not find videoUri in completed response")
                         
                    print(f"Video generated at: {video_uri}")
                    
                    # Download video
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                        output_path = tmp_file.name
                        
                    # Use GCS client to download
                    from google.cloud import storage
                    storage_client = storage.Client(project=self.project_id)
                    
                    if video_uri.startswith("gs://"):
                        parts = video_uri[5:].split("/", 1)
                        bucket_name = parts[0]
                        blob_name = parts[1]
                        
                        bucket = storage_client.bucket(bucket_name)
                        blob = bucket.blob(blob_name)
                        blob.download_to_filename(output_path)
                        
                        # Validate video file
                        if os.path.getsize(output_path) == 0:
                            raise ValueError("Generated video file is empty")
                            
                        print(f"Video saved to {output_path}")
                        return output_path
                    else:
                        raise ValueError(f"Unsupported video URI format: {video_uri}")
            
            raise TimeoutError("Video generation timed out after 10 minutes")

        except requests.exceptions.RequestException as e:
            print(f"Veo API request failed: {e}")
            if e.response:
                print(f"Response content: {e.response.text}")
            raise
        except Exception as e:
            print(f"Veo generation failed: {e}")
            raise
