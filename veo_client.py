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

    def _build_request_params(self, model_name: str, prompt: str) -> dict:
        """Build request parameters based on model version."""
        is_veo3 = model_name in self.VEO3_MODELS

        parameters = {
            "durationSeconds": 6,  # Valid for both Veo 2 (5-8) and Veo 3 (4,6,8)
            "aspectRatio": "16:9",
            "sampleCount": 1,
        }

        if is_veo3:
            # Veo 3 requires generateAudio, doesn't support enhancePrompt
            parameters["generateAudio"] = False
        else:
            # Veo 2 supports enhancePrompt
            parameters["enhancePrompt"] = True

        return {
            "instances": [{"prompt": prompt}],
            "parameters": parameters
        }

    def _try_generate_with_model(self, model_name: str, prompt: str) -> str:
        """
        Try to generate video with a specific model.
        Returns video path on success, raises exception on failure.
        """
        is_veo3 = model_name in self.VEO3_MODELS
        logger.info(f"Trying Veo model: {model_name} (Veo {'3.x' if is_veo3 else '2.x'})")

        url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{model_name}:predictLongRunning"
        payload = self._build_request_params(model_name, prompt)

        # Start the long-running operation
        response = requests.post(url, headers=self._get_headers(), json=payload, timeout=30)

        if response.status_code == 404:
            raise RuntimeError(f"Model {model_name} not found (404)")
        elif response.status_code == 403:
            raise RuntimeError(f"Access denied for model {model_name} (403)")
        elif response.status_code == 400:
            error_detail = response.json().get('error', {}).get('message', response.text)
            raise RuntimeError(f"Bad request for {model_name}: {error_detail}")

        response.raise_for_status()
        response_data = response.json()
        lro_name = response_data["name"]
        logger.info(f"✓ Video generation started with {model_name}: {lro_name}")

        # Poll for completion
        return self._poll_for_completion(lro_name, model_name)

    def _poll_for_completion(self, lro_name: str, model_name: str) -> str:
        """Poll the long-running operation until completion."""
        import tempfile
        import base64
        from google.cloud import storage

        start_time = time.time()
        poll_count = 0

        while time.time() - start_time < 600:  # 10 min timeout
            time.sleep(10)
            poll_count += 1

            poll_url = f"https://{self.region}-aiplatform.googleapis.com/v1/{lro_name}"
            poll_resp = requests.get(poll_url, headers=self._get_headers())
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()

            if poll_data.get("done"):
                if "error" in poll_data:
                    error_msg = poll_data['error']
                    raise RuntimeError(f"Video generation failed: {error_msg}")

                # Extract video from response
                response_obj = poll_data.get("response", {})

                # Try to get video data (base64 or GCS URI)
                video_base64 = None
                video_uri = None

                if "videos" in response_obj and response_obj["videos"]:
                    video_base64 = response_obj["videos"][0].get("bytesBase64Encoded")
                    video_uri = response_obj["videos"][0].get("gcsUri")
                elif "generatedSamples" in response_obj and response_obj["generatedSamples"]:
                    sample = response_obj["generatedSamples"][0]
                    if "video" in sample:
                        video_uri = sample["video"].get("gcsUri") or sample["video"].get("uri")

                # Create temp file for output
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                    output_path = tmp_file.name

                if video_base64:
                    logger.info(f"✓ Decoding base64 video from {model_name}")
                    video_bytes = base64.b64decode(video_base64)
                    with open(output_path, 'wb') as f:
                        f.write(video_bytes)
                    logger.info(f"✓ Video saved: {output_path} ({len(video_bytes)} bytes)")
                    return output_path

                elif video_uri and video_uri.startswith("gs://"):
                    logger.info(f"✓ Downloading video from {video_uri}")
                    storage_client = storage.Client(project=self.project_id)
                    parts = video_uri[5:].split("/", 1)
                    bucket = storage_client.bucket(parts[0])
                    blob = bucket.blob(parts[1])
                    blob.download_to_filename(output_path)

                    file_size = os.path.getsize(output_path)
                    if file_size == 0:
                        raise ValueError("Downloaded video is empty")
                    logger.info(f"✓ Video downloaded: {output_path} ({file_size} bytes)")
                    return output_path
                else:
                    raise ValueError(f"No video data in response: {response_obj}")

            elapsed = int(time.time() - start_time)
            logger.debug(f"Polling {model_name}... {elapsed}s elapsed")

        raise TimeoutError(f"Video generation timed out after 10 minutes")

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
        Generates a video using Veo, trying each model until one works.
        Returns the path to the saved video file.
        """
        logger.info(f"Starting Veo video generation with prompt: {prompt[:100]}...")

        errors = []
        for model_name in self.VEO_MODELS:
            try:
                video_path = self._try_generate_with_model(model_name, prompt)
                self.available_model = model_name  # Cache successful model
                return video_path
            except Exception as e:
                error_msg = str(e)
                errors.append(f"{model_name}: {error_msg}")
                logger.warning(f"✗ {model_name} failed: {error_msg}")

                # Don't try more models if it's a timeout (already waited 10 min)
                if "timed out" in error_msg.lower():
                    break
                continue

        # All models failed
        error_summary = "\n".join(errors)
        logger.error(f"All Veo models failed:\n{error_summary}")
        raise RuntimeError(f"Video generation failed with all models. Errors:\n{error_summary}")
