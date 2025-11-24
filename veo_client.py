import os
import time
import logging
import requests
import google.auth
import google.auth.transport.requests
from config import Config

logger = logging.getLogger(__name__)

class VeoClient:
    def __init__(self, project_id: str, region: str):
        if not project_id:
            raise ValueError("project_id is required for VeoClient")
        if not region:
            raise ValueError("region is required for VeoClient")

        self.project_id = project_id
        self.region = region
        # Base URL is set dynamically in generate_video based on the specific model version needed

        try:
            self.credentials, _ = google.auth.default()
            logger.info(f"VeoClient initialized for project {project_id} in region {region}")
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
        Generates a video using Veo 2.0, polls for completion, and saves to a temporary file.
        Returns the path to the saved video.
        """
        logger.info(f"Starting Veo video generation with prompt: {prompt[:100]}...")

        # Correct endpoint: predictLongRunning (not generateVideo)
        self.base_url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/veo-2.0-generate-001:predictLongRunning"

        # Correct payload format with instances and parameters
        payload = {
            "instances": [
                {
                    "prompt": prompt
                }
            ],
            "parameters": {
                "durationSeconds": 6,
                "aspectRatio": "16:9",
                "sampleCount": 1,
                "enhancePrompt": True
            }
        }

        try:
            # Start LRO
            logger.info(f"Sending request to: {self.base_url}")
            response = requests.post(
                self.base_url,
                headers=self._get_headers(),
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            response_data = response.json()
            lro_name = response_data["name"]
            logger.info(f"Video generation operation started: {lro_name}")

            # Poll for completion
            start_time = time.time()
            poll_count = 0
            while time.time() - start_time < 600:  # 10 min timeout
                time.sleep(10)  # Poll every 10s
                poll_count += 1

                poll_url = f"https://{self.region}-aiplatform.googleapis.com/v1/{lro_name}"
                logger.debug(f"Polling attempt {poll_count}: {poll_url}")
                poll_resp = requests.get(poll_url, headers=self._get_headers())
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()

                if "done" in poll_data and poll_data["done"]:
                    if "error" in poll_data:
                        error_msg = poll_data['error']
                        logger.error(f"Video generation failed with error: {error_msg}")
                        raise RuntimeError(f"Video generation failed: {error_msg}")

                    # Success! Parse response to get video URI
                    # Response format varies: check multiple possible locations
                    logger.debug(f"Operation complete. Response structure: {poll_data.keys()}")

                    video_uri = None
                    if "response" in poll_data:
                        response_obj = poll_data["response"]
                        # Try direct videoUri field
                        if "videoUri" in response_obj:
                            video_uri = response_obj["videoUri"]
                        # Try generatedSamples array format
                        elif "generatedSamples" in response_obj and response_obj["generatedSamples"]:
                            sample = response_obj["generatedSamples"][0]
                            if "video" in sample and "gcsUri" in sample["video"]:
                                video_uri = sample["video"]["gcsUri"]
                            elif "video" in sample and "uri" in sample["video"]:
                                video_uri = sample["video"]["uri"]

                    if not video_uri:
                        logger.error(f"Could not find video URI in response. Full response: {poll_data}")
                        raise ValueError("Could not find videoUri in completed response")

                    logger.info(f"Video successfully generated at: {video_uri}")
                    
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

                        logger.info(f"Downloading video from gs://{bucket_name}/{blob_name}")
                        bucket = storage_client.bucket(bucket_name)
                        blob = bucket.blob(blob_name)
                        blob.download_to_filename(output_path)

                        # Validate video file
                        file_size = os.path.getsize(output_path)
                        if file_size == 0:
                            raise ValueError("Generated video file is empty")

                        logger.info(f"Video successfully downloaded to {output_path} ({file_size} bytes)")
                        return output_path
                    else:
                        raise ValueError(f"Unsupported video URI format: {video_uri}")

                # Still polling - log progress
                elapsed = int(time.time() - start_time)
                logger.debug(f"Video generation in progress... {elapsed}s elapsed")

            raise TimeoutError(f"Video generation timed out after 10 minutes ({poll_count} polling attempts)")

        except requests.exceptions.RequestException as e:
            logger.error(f"Veo API request failed: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Veo video generation failed: {e}")
            raise
