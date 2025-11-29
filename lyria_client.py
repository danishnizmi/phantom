import os
import logging
import tempfile
import base64
import requests
import google.auth
import google.auth.transport.requests

logger = logging.getLogger(__name__)


class LyriaClient:
    """
    Client for generating audio/music using Google's Lyria model.
    See: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/lyria
    """

    # Lyria model
    MODEL_ID = "lyria-002"

    def __init__(self, project_id: str, region: str = "us-central1"):
        if not project_id:
            raise ValueError("project_id is required for LyriaClient")

        self.project_id = project_id
        self.region = region

        try:
            self.credentials, _ = google.auth.default()
            logger.info(f"LyriaClient initialized for project {project_id} in region {region}")
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

    def generate_audio(
        self,
        prompt: str,
        negative_prompt: str = None,
        seed: int = None,
        sample_count: int = 1
    ) -> str:
        """
        Generates instrumental music from a text prompt.

        Args:
            prompt: Text description of the audio to generate (e.g., "An energetic electronic dance track")
            negative_prompt: Optional description of what to exclude (e.g., "vocals, slow tempo")
            seed: Optional seed for deterministic generation (cannot be used with sample_count > 1)
            sample_count: Number of audio samples to generate (1-4, cannot be used with seed)

        Returns:
            Path to the generated WAV file (30 seconds, 48kHz)
        """
        logger.info(f"Starting Lyria audio generation with prompt: {prompt[:100]}...")

        url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.MODEL_ID}:predict"

        # Build instance
        instance = {"prompt": prompt}
        if negative_prompt:
            instance["negative_prompt"] = negative_prompt
        if seed is not None:
            instance["seed"] = seed

        # Build parameters
        parameters = {}
        if sample_count > 1 and seed is None:
            parameters["sample_count"] = sample_count

        payload = {
            "instances": [instance],
            "parameters": parameters
        }

        try:
            logger.info(f"Sending request to Lyria API...")
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=120  # Audio generation can take a while
            )

            if response.status_code == 404:
                raise RuntimeError(f"Lyria model {self.MODEL_ID} not found (404)")
            elif response.status_code == 403:
                raise RuntimeError(f"Access denied for Lyria model (403)")
            elif response.status_code == 400:
                error_detail = response.json().get('error', {}).get('message', response.text)
                raise RuntimeError(f"Bad request: {error_detail}")

            response.raise_for_status()
            response_data = response.json()

            # Extract audio from predictions
            predictions = response_data.get("predictions", [])
            if not predictions:
                raise ValueError("No audio generated in response")

            # Get the first audio clip
            audio_data = predictions[0]
            audio_base64 = audio_data.get("audioContent")
            mime_type = audio_data.get("mimeType", "audio/wav")

            if not audio_base64:
                raise ValueError("No audioContent in response")

            # Decode and save
            audio_bytes = base64.b64decode(audio_base64)

            # Determine file extension
            ext = ".wav" if "wav" in mime_type else ".mp3"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                output_path = tmp_file.name

            with open(output_path, 'wb') as f:
                f.write(audio_bytes)

            logger.info(f"✓ Audio generated: {output_path} ({len(audio_bytes)} bytes, {mime_type})")
            return output_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Lyria API request failed: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Lyria audio generation failed: {e}")
            raise

    def generate_background_music(self, mood: str = "tech", energy: str = "medium") -> str:
        """
        Convenience method to generate background music for videos.

        Args:
            mood: The mood of the music (tech, chill, dramatic, upbeat)
            energy: Energy level (low, medium, high)

        Returns:
            Path to the generated WAV file
        """
        mood_prompts = {
            "tech": "Modern electronic ambient music with synthesizers and subtle beats",
            "chill": "Calm lo-fi hip hop beats with soft piano and mellow vibes",
            "dramatic": "Cinematic orchestral music with building tension and epic strings",
            "upbeat": "Energetic electronic dance music with driving beats and synths",
        }

        energy_modifiers = {
            "low": "slow tempo, minimal, relaxed",
            "medium": "moderate tempo, balanced",
            "high": "fast tempo, intense, energetic",
        }

        base_prompt = mood_prompts.get(mood, mood_prompts["tech"])
        energy_mod = energy_modifiers.get(energy, energy_modifiers["medium"])

        prompt = f"{base_prompt}, {energy_mod}, instrumental, no vocals, suitable for background"
        negative_prompt = "vocals, singing, spoken words, harsh noise"

        return self.generate_audio(prompt=prompt, negative_prompt=negative_prompt)
