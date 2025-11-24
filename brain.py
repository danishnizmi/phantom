import logging
import vertexai
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview.generative_models import grounding
from vertexai.preview.vision_models import ImageGenerationModel
from google.cloud import firestore
from config import Config
import datetime
import os
import re
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentBrain:
    def __init__(self):
        self.project_id = Config.PROJECT_ID
        self.location = Config.REGION
        
        vertexai.init(project=self.project_id, location=self.location)
        
        # Multi-model configuration with dynamic discovery
        # Try Gemini 1.5 first for search compatibility, then newer models
        candidate_models = [
            "gemini-1.5-pro",          # Best for search grounding (if available)
            "gemini-1.5-flash",        # Fast, supports search (if available)
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite-001",
            "gemini-2.5-pro",
        ]
        
        # Dynamic Discovery: Try to fetch models from GCP (e.g. tuned models)
        try:
            aiplatform.init(project=self.project_id, location=self.location)
            # List models with "gemini" in the name
            gcp_models = aiplatform.Model.list(filter='display_name:"gemini*"')
            for m in gcp_models:
                # Use resource name or display name
                # Foundation models might not appear here, but tuned ones will
                if m.display_name not in candidate_models:
                    candidate_models.append(m.display_name)
                    logger.info(f"Discovered GCP model: {m.display_name}")
        except Exception as e:
            logger.warning(f"Could not list GCP models (using default candidates): {e}")
        
        self.model_names = []
        self.models = {}
        
        logger.info("Discovering available Gemini models...")
        
        # Test each candidate model to see if it's available
        for model_name in candidate_models:
            try:
                model = GenerativeModel(model_name)
                # Quick test to verify model is accessible
                test_response = model.generate_content("Hi", 
                    generation_config={"max_output_tokens": 5, "temperature": 0})
                
                if test_response.text:
                    self.models[model_name] = model
                    self.model_names.append(model_name)
                    logger.info(f"✓ Verified model: {model_name}")
                else:
                    logger.warning(f"⚠ {model_name} responded but empty")
                    
            except Exception as e:
                error_str = str(e)
                if "404" in error_str:
                    logger.debug(f"✗ {model_name} not available (404)")
                else:
                    logger.warning(f"✗ {model_name} failed: {error_str[:80]}")
        
        if not self.models:
            raise RuntimeError(f"No Gemini models available. Tried: {candidate_models}")

        logger.info(f"✓ Active models ({len(self.models)}): {self.model_names}")

        # Initialize Google Search Grounding Tool (only if 1.5 models available)
        # NOTE: Gemini 2.0/2.5 don't support google_search_retrieval yet
        # See: https://github.com/GoogleCloudPlatform/generative-ai/issues/667
        self.search_tool = None
        has_1_5_model = any("1.5" in name for name in self.model_names)

        if has_1_5_model:
            try:
                self.search_tool = Tool.from_google_search_retrieval(
                    google_search_retrieval=grounding.GoogleSearchRetrieval()
                )
                logger.info("✓ Google Search grounding enabled (Gemini 1.5 models available)")
            except Exception as e:
                logger.warning(f"Could not initialize search tool: {e}")
        else:
            logger.warning("⚠ Google Search grounding disabled (Gemini 1.5 models not available)")
            logger.warning("   Using instructed search mode - model will be told to only use real URLs")

        self.db = firestore.Client(project=self.project_id)
        self.collection = self.db.collection(Config.COLLECTION_NAME)

    def _extract_urls(self, text: str) -> list:
        """Extracts all URLs from text."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.findall(url_pattern, text)

    def _validate_url(self, url: str) -> bool:
        """Basic URL validation - checks if it's not obviously fake."""
        if not url or len(url) < 10:
            return False

        # Check for common fake URL patterns
        fake_patterns = [
            'example.com',
            'placeholder',
            'yoursite.com',
            'tempurl',
            'fake',
        ]

        url_lower = url.lower()
        for pattern in fake_patterns:
            if pattern in url_lower:
                return False

        # Must have a valid TLD
        if not re.search(r'\.(com|org|net|edu|gov|io|ai|dev|tech|co|info|blog)(/|$)', url_lower):
            return False

        return True

    def _generate_with_fallback(self, prompt: str, tools: list = None, require_url: bool = False) -> str:
        """
        Attempts to generate content using available models with fallback.
        Tries each model in order until successful or all fail.

        If require_url=True and tools are provided, validates that response contains real URLs.
        """
        last_error = None

        for model_name in self.model_names:
            if model_name not in self.models:
                continue

            try:
                model = self.models[model_name]
                # Pass tools if provided (e.g. Grounding)
                response = model.generate_content(prompt, tools=tools)

                if response.text:
                    text = response.text.strip()

                    # If URL validation is required
                    if require_url and tools:
                        urls = self._extract_urls(text)
                        valid_urls = [url for url in urls if self._validate_url(url)]

                        if not valid_urls:
                            logger.warning(f"✗ {model_name} generated content without valid URLs, trying next model")
                            continue

                        logger.info(f"✓ Generated content with {model_name} (found {len(valid_urls)} valid URLs)")
                    else:
                        logger.info(f"✓ Generated content with {model_name}")

                    return text
                else:
                    logger.warning(f"✗ {model_name} returned empty response")

            except Exception as e:
                last_error = e
                logger.warning(f"✗ {model_name} failed: {str(e)[:100]}")
                continue

        # All models failed
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def _get_trending_topic(self) -> str:
        """
        Asks Gemini to identify a trending tech topic.
        Uses Google Search if available, otherwise relies on model's recent knowledge.
        """
        if self.search_tool:
            prompt = """Find the single most interesting tech news story RIGHT NOW (last 24 hours).
            Focus on:
            - Major AI product launches or updates
            - Significant open source releases
            - Big tech industry moves

            Return ONLY the headline/topic name. Be specific."""
            # Use search tool to get real-time info
            return self._generate_with_fallback(prompt, tools=[self.search_tool])
        else:
            # Without search: ask for recent topics from model's knowledge
            prompt = """What is a significant tech news or product launch from your recent knowledge?
            Focus on:
            - Major AI products (ChatGPT, Claude, Gemini, etc.)
            - Popular open source projects (React, Next.js, Tailwind, etc.)
            - Big tech companies (Google, Microsoft, Meta, Apple, etc.)

            Return ONLY a specific product/project name and brief context. Be real and specific.
            Example: "Anthropic Claude 3 Opus" or "Meta Llama 3" or "OpenAI GPT-4 Turbo"

            Do NOT make up fake products or news."""
            return self._generate_with_fallback(prompt)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _check_history(self, topic: str) -> bool:
        """
        Checks Firestore to see if we've recently posted about this topic.
        Returns True if we should SKIP this topic (duplicate), False otherwise.
        """
        try:
            # Check last 20 posts
            docs = self.collection.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20).stream()
            
            def _normalize(text):
                return set(text.lower().split())
            
            current_words = _normalize(topic)
            
            for doc in docs:
                data = doc.to_dict()
                stored_topic = data.get("topic", "")
                stored_words = _normalize(stored_topic)
                
                # Keyword overlap check (Jaccard similarity)
                if not current_words: continue
                overlap = len(stored_words & current_words) / len(current_words)
                
                if overlap > 0.6: # 60% overlap
                    logger.info(f"Topic '{topic}' matches recent post '{stored_topic}' (Overlap: {overlap:.2f}). Skipping.")
                    return True
            return False
        except Exception as e:
            logger.warning(f"Firestore history check failed: {e}. Proceeding without check.")
            return False

    def generate_image(self, prompt: str) -> str:
        """
        Generates an image using Imagen 3 Fast and saves it to a temp file.
        Returns the path to the saved image.
        """
        logger.info(f"Generating image for: {prompt}")
        try:
            # Switch to Fast model for speed
            model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")
            images = model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio="16:9",
                safety_filter_level="block_some",
                person_generation="allow_adult"
            )
            
            if not images:
                raise ValueError("No images generated")
                
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                output_path = tmp_file.name
                
            images[0].save(location=output_path, include_generation_parameters=False)
            logger.info(f"Image saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Imagen generation failed: {e}")
            raise

    def get_strategy(self):
        """
        Decides on the content strategy: Text (HN Style), Video (Veo), or Image (Imagen).
        Returns a dict with 'type', 'content', 'topic', and optional 'video_prompt'/'image_path'.
        """
        topic = self._get_trending_topic()
        
        # Retry logic for duplicates
        if self._check_history(topic):
            logger.info("Duplicate topic detected. Requesting alternative.")
            prompt = f"""The topic '{topic}' was already covered. Find a DIFFERENT trending tech news story from the last 24 hours.
                Return ONLY the headline."""
            try:
                new_topic = self._generate_with_fallback(prompt, tools=[self.search_tool])
                if self._check_history(new_topic):
                     # Try one more time with explicit instruction
                     prompt2 = f"Both '{topic}' and '{new_topic}' are taken. Give me a completely random but interesting tech tool or library name."
                     new_topic = self._generate_with_fallback(prompt2)
                     
                topic = new_topic
            except Exception as e:
                logger.error(f"Failed to find alternative topic: {e}")
                # Proceed with original topic if fallback fails, better than crashing
        
        logger.info(f"Selected Topic: {topic}")

        # Decide format
        if Config.BUDGET_MODE:
            logger.info("BUDGET_MODE enabled, using text-only format")
            post_type = "text"
        else:
            logger.info("BUDGET_MODE disabled, deciding optimal format for media generation")
            # Ask Gemini if this topic is better for video, image, or text
            decision_prompt = f"""For the tech news '{topic}', what is the BEST visual format?

Consider:
- VIDEO: Product demos, UI animations, dynamic visualizations, tutorials
- IMAGE: Product launches, devices, infographics, architecture diagrams, logos, concepts
- TEXT: Pure text news, policy changes, financial updates, abstract concepts

IMPORTANT: Prefer VIDEO or IMAGE when possible for engagement. Only choose TEXT if the topic is truly text-heavy or abstract.

Reply with EXACTLY ONE WORD: VIDEO, IMAGE, or TEXT"""

            try:
                decision = self._generate_with_fallback(decision_prompt).upper()
                logger.info(f"Format decision response: {decision}")

                if "VIDEO" in decision:
                    post_type = "video"
                elif "IMAGE" in decision:
                    post_type = "image"
                else:
                    post_type = "text"

                logger.info(f"Selected post type: {post_type}")
            except Exception as e:
                logger.warning(f"Format decision failed: {e}, defaulting to text")
                post_type = "text"

        strategy = {
            "topic": topic,
            "type": post_type,
            "timestamp": firestore.SERVER_TIMESTAMP
        }

        if post_type == "video":
            # Generate Video Prompt and Tweet Text
            script_prompt = f"Write a tweet caption for a video about '{topic}'. Also provide a visual prompt for an AI video generator. Format: CAPTION: <text> | PROMPT: <visual description>"
            try:
                response = self._generate_with_fallback(script_prompt)
                if "|" in response:
                    parts = response.split("|")
                    caption = parts[0].replace("CAPTION:", "").strip()
                    visual_prompt = parts[1].replace("PROMPT:", "").strip()
                else:
                    caption = response[:100]
                    visual_prompt = f"Tech visualization of {topic}"
            except Exception as e:
                logger.error(f"Failed to generate video script: {e}")
                raise 
            
            strategy["content"] = caption
            strategy["video_prompt"] = visual_prompt
            
        elif post_type == "image":
            # Generate Image Prompt and Tweet Text
            script_prompt = f"Write a tweet caption for an image about '{topic}'. Also provide a visual prompt for an AI image generator (Imagen). Format: CAPTION: <text> | PROMPT: <visual description>"
            try:
                response = self._generate_with_fallback(script_prompt)
                if "|" in response:
                    parts = response.split("|")
                    caption = parts[0].replace("CAPTION:", "").strip()
                    visual_prompt = parts[1].replace("PROMPT:", "").strip()
                else:
                    caption = response[:100]
                    visual_prompt = f"High quality tech photography of {topic}"
            except Exception as e:
                logger.error(f"Failed to generate image script: {e}")
                raise 
                
            strategy["content"] = caption
            strategy["image_prompt"] = visual_prompt
            
        else:
            # Generate Hacker News Style Post
            logger.info(f"Generating HN-style post for: {topic}")

            if self.search_tool:
                # With search grounding - can find real URLs
                post_prompt = f"""Search Google for recent news about '{topic}' and write an engaging developer-focused tweet.

CRITICAL REQUIREMENTS:
1. You MUST use Google Search to find a REAL, working URL about this topic
2. The URL MUST be from an actual news site, blog, or official announcement
3. Include the complete URL in your response (starting with https://)
4. DO NOT make up or hallucinate URLs
5. Write in this exact format:

[Bold claim or question about the news]

[REAL URL you found via search]

[Provocative insight that invites discussion]

CONSTRAINTS:
- Total length: Under 280 characters
- NO hashtags, NO emojis
- If you cannot find a real URL, say "No URL found" instead of making one up
"""
                try:
                    response = self._generate_with_fallback(
                        post_prompt,
                        tools=[self.search_tool],
                        require_url=True
                    )
                    tweet = response.strip()
                except Exception as e:
                    logger.error(f"Search-based generation failed: {e}, falling back to no-URL mode")
                    self.search_tool = None  # Disable for this run

            if not self.search_tool:
                # Without search grounding - provide known/example URLs
                logger.info("Generating post WITHOUT URL (search grounding unavailable)")
                post_prompt = f"""Write an engaging developer-focused tweet about '{topic}'.

CRITICAL INSTRUCTIONS:
1. DO NOT include any URLs - we cannot verify them
2. Focus on the technology and its impact
3. Ask a provocative question or make an interesting observation
4. Be authentic and don't overhype

Format:
[Interesting fact or question about {topic}]

[Technical insight or opinion that invites discussion]

CONSTRAINTS:
- Total length: Under 280 characters
- NO hashtags, NO emojis, NO URLs
- Be truthful and don't make up facts

Example style: "The new model outperforms GPT-4 on reasoning tasks. But can it actually replace human developers? Probably not yet."
"""
                try:
                    response = self._generate_with_fallback(post_prompt)
                    tweet = response.strip()
                except Exception as e:
                    logger.error(f"Failed to generate post: {e}")
                    raise

            # Strict length check
            if len(tweet) > 280:
                logger.warning(f"Tweet too long ({len(tweet)}), truncating.")
                tweet = tweet[:277] + "..."

            # Final validation
            if not tweet or len(tweet) < 10:
                tweet = f"{topic} is making waves in tech. What's your take?"

            strategy["content"] = [tweet]

        return strategy

    def log_post(self, strategy: dict, success: bool, error: str = None):
        """Logs the attempt to Firestore."""
        try:
            doc_ref = self.collection.document()
            data = strategy.copy()
            data["success"] = success
            if error:
                data["error"] = error
            if "timestamp" not in data:
                data["timestamp"] = firestore.SERVER_TIMESTAMP
            doc_ref.set(data)
        except Exception as e:
            logger.error(f"Failed to log post to Firestore: {e}")
