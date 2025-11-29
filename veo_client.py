import os
import time
import logging
import requests
import google.auth
import google.auth.transport.requests
from config import Config

logger = logging.getLogger(__name__)

class VeoClient:
    # Veo model variants to try (in order of preference)
    # See: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation
    VEO_MODELS = [
        "veo-3.1-fast-generate-001",  # Veo 3.1 Fast - newest
        "veo-3.1-generate-001",       # Veo 3.1 - highest quality
        "veo-3.0-fast-generate-001",  # Veo 3 Fast - best latency
        "veo-3.0-generate-001",       # Veo 3 - high quality
        "veo-2.0-generate-001",       # Veo 2 GA
        "veo-2.0-generate-exp",       # Veo 2 Experimental
    ]

    # Models that are Veo 3.x (require generateAudio, don't support enhancePrompt)
    VEO3_MODELS = {"veo-3.0-fast-generate-001", "veo-3.0-generate-001",
                   "veo-3.1-fast-generate-001", "veo-3.1-generate-001",
                   "veo-3.1-generate-preview", "veo-3.1-fast-generate-preview"}

    def __init__(self, project_id: str, region: str):
        if not project_id:
            raise ValueError("project_id is required for VeoClient")
        if not region:
            raise ValueError("region is required for VeoClient")

        self.project_id = project_id
        self.region = region
        self.available_model = None

        try:
            self.credentials, _ = google.auth.default()
            logger.info(f"VeoClient initialized for project {project_id} in region {region}")
        except Exception as e:
            raise RuntimeError(f"Failed to get default credentials: {e}")

    def _check_model_available(self, model_name: str) -> bool:
        """Check if a Veo model is available by querying the API."""
        try:
            check_url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{model_name}"
            response = requests.get(check_url, headers=self._get_headers(), timeout=10)
            if response.status_code == 200:
                logger.info(f"✓ Veo model available: {model_name}")
                return True
            else:
                logger.debug(f"✗ Veo model not available: {model_name} ({response.status_code})")
                return False
        except Exception as e:
            logger.debug(f"✗ Veo model check failed: {model_name} - {e}")
            return False

    def get_available_model(self) -> str:
        """Find first available Veo model."""
        if self.available_model:
            return self.available_model

        for model in self.VEO_MODELS:
            if self._check_model_available(model):
                self.available_model = model
                return model

        raise RuntimeError(f"No Veo models available. Tried: {self.VEO_MODELS}")

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
        Dynamically discovers available Veo model.
        Returns the path to the saved video.
        """
        logger.info(f"Starting Veo video generation with prompt: {prompt[:100]}...")

        # Dynamically find available Veo model
        model_name = self.get_available_model()
        logger.info(f"Using Veo model: {model_name}")

        # Correct endpoint: predictLongRunning for video generation
        self.base_url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{model_name}:predictLongRunning"

        # Build parameters based on model version (Veo 2 vs Veo 3)
        is_veo3 = model_name in self.VEO3_MODELS
        parameters = {
            "durationSeconds": 6,  # Valid for both Veo 2 (5-8) and Veo 3 (4,6,8)
            "aspectRatio": "16:9",
            "sampleCount": 1,
        }

        if is_veo3:
            # Veo 3 requires generateAudio, doesn't support enhancePrompt
            parameters["generateAudio"] = False  # No audio needed for social media
            logger.info(f"Using Veo 3 parameters (generateAudio=False)")
        else:
            # Veo 2 supports enhancePrompt
            parameters["enhancePrompt"] = True
            logger.info(f"Using Veo 2 parameters (enhancePrompt=True)")

        payload = {
            "instances": [
                {
                    "prompt": prompt
                }
            ],
            "parameters": parameters
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
                    # Response format per docs: response.videos[].gcsUri
                    logger.debug(f"Operation complete. Response structure: {poll_data.keys()}")

                    video_uri = None
                    if "response" in poll_data:
                        response_obj = poll_data["response"]
                        # Primary format per docs: videos array with gcsUri
                        if "videos" in response_obj and response_obj["videos"]:
                            video_uri = response_obj["videos"][0].get("gcsUri")
                        # Fallback: direct videoUri field
                        elif "videoUri" in response_obj:
                            video_uri = response_obj["videoUri"]
                        # Fallback: generatedSamples array format (older API)
                        elif "generatedSamples" in response_obj and response_obj["generatedSamples"]:
                            sample = response_obj["generatedSamples"][0]
                            if "video" in sample and "gcsUri" in sample["video"]:
                                video_uri = sample["video"]["gcsUri"]
                            elif "video" in sample and "uri" in sample["video"]:
                                video_uri = sample["video"]["uri"]

                    # Check for base64 encoded video first (when no storageUri provided)
                    video_base64 = None
                    if "response" in poll_data:
                        response_obj = poll_data["response"]
                        if "videos" in response_obj and response_obj["videos"]:
                            video_base64 = response_obj["videos"][0].get("bytesBase64Encoded")

                    # Download video
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                        output_path = tmp_file.name

                    if video_base64:
                        # Decode base64 video directly
                        import base64
                        logger.info("Decoding base64 encoded video from response")
                        video_bytes = base64.b64decode(video_base64)
                        with open(output_path, 'wb') as f:
                            f.write(video_bytes)
                        file_size = len(video_bytes)
                        logger.info(f"Video saved to {output_path} ({file_size} bytes)")
                        return output_path

                    elif video_uri:
                        logger.info(f"Video generated at: {video_uri}")

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
                    else:
                        logger.error(f"Could not find video in response. Full response: {poll_data}")
                        raise ValueError("Could not find video data in completed response")

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
