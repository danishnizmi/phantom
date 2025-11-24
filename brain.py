import logging
import vertexai
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview.generative_models import grounding
from vertexai.preview.vision_models import ImageGenerationModel
from google.cloud import firestore
from config import Config
from news_fetcher import NewsFetcher
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
                    logger.warning(f"WARNING {model_name} responded but empty")
                    
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
            logger.warning("WARNING Google Search grounding disabled (Gemini 1.5 models not available)")
            logger.warning("   Using instructed search mode - model will be told to only use real URLs")

        self.db = firestore.Client(project=self.project_id)
        self.collection = self.db.collection(Config.COLLECTION_NAME)

        # Initialize news fetcher for real URLs
        self.news_fetcher = NewsFetcher()
        logger.info("✓ News fetcher initialized (Hacker News API)")

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

    def _get_trending_story(self) -> dict:
        """
        Gets a trending tech story with REAL URL from Hacker News or other sources.
        Returns dict with {title, url, source}.
        """
        story = self.news_fetcher.get_trending_story()

        if story:
            logger.info(f"✓ Found trending story: {story['title'][:50]}...")
            return story

        # Fallback: use model to suggest a topic (no URL)
        logger.warning("Could not fetch real news, falling back to model knowledge")
        prompt = """Suggest ONE specific, real tech product or project that developers would find interesting.
        Examples: "Next.js 15", "Anthropic Claude 3.5 Sonnet", "Meta Llama 3"

        Return ONLY the name. Be specific and real."""

        topic_name = self._generate_with_fallback(prompt)
        return {
            'title': topic_name,
            'url': None,  # No URL available
            'source': 'model_knowledge'
        }

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

    def _validate_strategy(self, strategy: dict) -> dict:
        """
        Final validation check - AI reviews the strategy to ensure it makes sense.
        Returns dict with {'valid': bool, 'reason': str}
        """
        post_type = strategy.get('type')
        topic = strategy.get('topic')
        content = strategy.get('content')
        source_url = strategy.get('source_url')
        category = strategy.get('category', 'unknown')

        # Build validation prompt with STRICT reality checks
        content_text = content[0] if isinstance(content, list) else content
        validation_prompt = f"""You are a STRICT quality control AI. Your job is to REJECT fake, made-up, or misleading content.

ACTUAL NEWS STORY:
- Real Topic: "{topic}"
- Real URL: {source_url if source_url else 'No URL provided'}
- Category: {category}

GENERATED CONTENT TO VALIDATE:
- Type: {post_type}
- Content: "{content_text}"

CRITICAL VALIDATION - BE VERY STRICT:

1. ✓ Does content relate to the ACTUAL news topic "{topic}"?
   - REJECT if it talks about different products/companies
   - REJECT if it mentions products not in the topic
   - REJECT if it's promotional/marketing language

2. ✓ Is content FACTUAL and not made-up?
   - REJECT if it mentions fake product names (e.g., "Nano Banana Pro")
   - REJECT if it claims features not mentioned in topic
   - REJECT if it sounds like an advertisement
   - REJECT if it invents version numbers or specs

3. ✓ Is content COMPLETE?
   - Must end with punctuation (. ! ?)
   - Must be full sentences
   - Character count: {len(content_text)} (must be 20-280)

4. ✓ Is tone appropriate?
   - Engaging but NOT promotional
   - Question-based or provocative, NOT hype
   - Developer/tech audience, NOT consumer marketing

EXAMPLES OF WHAT TO REJECT:
- "Unleash your creative vision with [product]" - Marketing language
- Mentions products not in the original topic - Made up
- "Check back later for video/updates!" - Placeholder text
- Any "Pro" or version numbers not in topic - Fabricated
- News posts without URLs when URL is available - Missing citation

EXAMPLES OF WHAT TO APPROVE:
- "New AI model. Can it handle production?\n\nhttps://..." - Has URL
- "[Real product] launches today. Will devs use it?" - Real, engaging

DECISION (BE STRICT - WHEN IN DOUBT, REJECT):
Reply EXACTLY:

APPROVE: [reason it matches topic AND is factual]
OR
REJECT: [specific problem - made up content, wrong topic, promotional, etc.]

Now validate: "{content_text}"
Does it relate to actual topic "{topic}"? Are all claims real?
"""

        try:
            response = self._generate_with_fallback(validation_prompt)
            response_upper = response.upper()

            logger.info(f"Validation response: {response}")

            if 'APPROVE' in response_upper:
                reason = response.split(':', 1)[1].strip() if ':' in response else response
                return {'valid': True, 'reason': reason}
            elif 'REJECT' in response_upper:
                reason = response.split(':', 1)[1].strip() if ':' in response else response
                return {'valid': False, 'reason': reason}
            else:
                # Unclear response, be strict - reject
                logger.warning(f"Unclear validation response: {response}")
                return {'valid': False, 'reason': f"Validation unclear, rejecting to be safe: {response}"}

        except Exception as e:
            logger.error(f"Validation check failed: {e}")
            # If validation itself fails, REJECT to be safe (don't post potentially bad content)
            return {'valid': False, 'reason': f"Validation system error, rejecting for safety: {e}"}

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
        max_retries = 3

        for attempt in range(max_retries):
            try:
                return self._generate_strategy_with_validation(attempt)
            except ValueError as e:
                if "validation failed" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed validation: {e}")
                    logger.info(f"Retrying with different story... ({attempt + 2}/{max_retries})")
                    continue
                else:
                    raise

        raise RuntimeError("Failed to generate valid strategy after all retries")

    def _generate_strategy_with_validation(self, attempt: int = 0) -> dict:
        """
        Internal method to generate and validate a strategy.
        Separated for retry logic.
        """
        story = self._get_trending_story()
        topic = story['title']
        story_url = story.get('url')  # Real URL or None

        # Retry logic for duplicates
        if self._check_history(topic):
            logger.info("Duplicate topic detected. Requesting alternative.")
            # Try to get a different story
            try:
                for _ in range(3):  # Try up to 3 times
                    story = self._get_trending_story()
                    topic = story['title']
                    story_url = story.get('url')

                    if not self._check_history(topic):
                        break
                else:
                    logger.warning("All alternatives were duplicates, proceeding with latest")
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
            script_prompt = f"""Generate a tweet with video for THIS EXACT TOPIC: '{topic}'

CRITICAL WARNING: You MUST write about THIS EXACT topic. DO NOT make up fake products or features!

You MUST provide BOTH parts in this EXACT format:
CAPTION: <your complete tweet text here>
PROMPT: <your visual description here>

CAPTION REQUIREMENTS:
- Must reference the ACTUAL topic: '{topic}'
- Do NOT invent product names, versions, or features
- Must be a COMPLETE sentence ending with punctuation (. ! ?)
- 100-200 characters total
- Engaging question or observation for developers
- NO marketing language ("Unleash", "Revolutionary", etc.)
- NO hashtags, NO emojis

PROMPT REQUIREMENTS:
- Visual description for video generator
- Tech-focused, developer-oriented visuals
- 50-100 characters

BAD Examples (NEVER DO THIS):
X "Unleash creativity with Nano Banana Pro!"
X "Check back later for the video!"
X Any filler text about waiting

GOOD Examples:
OK "New AI model from Google. Can it replace developers?"
OK "Latest chip promises 10x speedup. Will it deliver?"

CRITICAL: Write COMPLETE, STANDALONE caption. NO placeholder text!

Now generate for: '{topic}'
"""

            try:
                response = self._generate_with_fallback(script_prompt)
                logger.info(f"Video generation response: {response[:100]}...")

                if "CAPTION:" in response and "PROMPT:" in response:
                    parts = response.split("PROMPT:")
                    caption_part = parts[0].replace("CAPTION:", "").strip()
                    visual_prompt = parts[1].strip()

                    # Validate caption is complete
                    if len(caption_part) < 20 or not any(caption_part.endswith(p) for p in ['.', '!', '?']):
                        logger.warning(f"Caption seems incomplete: {caption_part}")
                        caption_part = f"{caption_part}. What's your take?"

                    caption = caption_part[:200]  # Enforce max length
                else:
                    logger.error("Response missing CAPTION: or PROMPT: markers")
                    raise ValueError("Invalid format - missing CAPTION or PROMPT")

            except Exception as e:
                logger.error(f"Failed to generate video script: {e}")
                raise

            # For video posts with URL, add URL to caption for citation
            if story_url:
                if story_url not in caption:
                    # Add URL to caption
                    if len(caption) + len(story_url) + 4 <= 200:  # 4 for "\n\n"
                        caption = f"{caption}\n\n{story_url}"
                    else:
                        # Truncate caption to fit URL
                        max_cap_len = 200 - len(story_url) - 7  # 7 for "...\n\n"
                        caption = f"{caption[:max_cap_len]}...\n\n{story_url}"
                    logger.info(f"Added URL to video caption: {story_url}")

            strategy["content"] = caption
            strategy["video_prompt"] = visual_prompt

        elif post_type == "image":
            # Generate Image Prompt and Tweet Text
            script_prompt = f"""Generate a tweet with image for THIS EXACT TOPIC: '{topic}'

CRITICAL WARNING: You MUST write about THIS EXACT topic. DO NOT make up fake products or features!

You MUST provide BOTH parts in this EXACT format:
CAPTION: <your complete tweet text here>
PROMPT: <your visual description here>

CAPTION REQUIREMENTS:
- Must reference the ACTUAL topic: '{topic}'
- Do NOT invent product names, versions, or features
- Must be a COMPLETE sentence ending with punctuation (. ! ?)
- 100-200 characters total
- Engaging question or observation for developers
- NO marketing language ("Unleash", "Revolutionary", etc.)
- NO hashtags, NO emojis

PROMPT REQUIREMENTS:
- Visual description for image generator (Imagen)
- Tech photography or illustration style
- 50-100 characters

BAD Examples (NEVER DO THIS):
X "Unleash creativity with Gemini 3 Pro Image!"
X "Check back later for updates!"
X Any placeholder or filler text

GOOD Examples:
OK "OpenAI's new vision model. Can it debug CSS layouts?"
OK "AI chip promises 10x gains. But at what cost?"

CRITICAL: Write COMPLETE, STANDALONE caption. NO placeholder text!

Now generate for: '{topic}'
"""

            try:
                response = self._generate_with_fallback(script_prompt)
                logger.info(f"Image generation response: {response[:100]}...")

                if "CAPTION:" in response and "PROMPT:" in response:
                    parts = response.split("PROMPT:")
                    caption_part = parts[0].replace("CAPTION:", "").strip()
                    visual_prompt = parts[1].strip()

                    # Validate caption is complete
                    if len(caption_part) < 20 or not any(caption_part.endswith(p) for p in ['.', '!', '?']):
                        logger.warning(f"Caption seems incomplete: {caption_part}")
                        caption_part = f"{caption_part}. What's your take?"

                    caption = caption_part[:200]  # Enforce max length
                else:
                    logger.error("Response missing CAPTION: or PROMPT: markers")
                    raise ValueError("Invalid format - missing CAPTION or PROMPT")

            except Exception as e:
                logger.error(f"Failed to generate image script: {e}")
                raise

            # For image posts with URL, add URL to caption for citation
            if story_url:
                if story_url not in caption:
                    # Add URL to caption
                    if len(caption) + len(story_url) + 4 <= 200:  # 4 for "\n\n"
                        caption = f"{caption}\n\n{story_url}"
                    else:
                        # Truncate caption to fit URL
                        max_cap_len = 200 - len(story_url) - 7  # 7 for "...\n\n"
                        caption = f"{caption[:max_cap_len]}...\n\n{story_url}"
                    logger.info(f"Added URL to image caption: {story_url}")

            strategy["content"] = caption
            strategy["image_prompt"] = visual_prompt

        else:
            # Generate Hacker News Style Post with REAL URL
            logger.info(f"Generating HN-style post for: {topic}")

            if story_url:
                # We have a REAL URL from news fetcher!
                logger.info(f"Using real URL: {story_url}")
                post_prompt = f"""Write an engaging developer-focused tweet about this story:

Title: "{topic}"
URL: {story_url}

Format:
[Bold claim or provocative question about the news]

{story_url}

[Technical insight that makes developers want to reply]

CONSTRAINTS:
- Total length: Under 280 characters (including the URL above)
- Use the EXACT URL provided above
- NO hashtags, NO emojis
- Make it engaging and slightly provocative

Example style: "Finally, a framework that doesn't need 47 config files to start. But will it scale?"
"""
            else:
                # No URL available - text-only tweet
                logger.info("No URL available, generating text-only post")
                post_prompt = f"""Write an engaging developer-focused tweet about '{topic}'.

CRITICAL INSTRUCTIONS:
1. DO NOT include any URLs - we don't have one
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

                # If we have a URL, ensure it's in the tweet
                if story_url and story_url not in tweet:
                    logger.warning("Generated tweet missing URL, adding it")
                    # Try to fit URL in
                    max_text_len = 280 - len(story_url) - 2  # -2 for spacing
                    if len(tweet) > max_text_len:
                        tweet = tweet[:max_text_len-3] + "..."
                    tweet = f"{tweet}\n\n{story_url}"

            except Exception as e:
                logger.error(f"Failed to generate post: {e}")
                raise

            # Strict length check
            if len(tweet) > 280:
                logger.warning(f"Tweet too long ({len(tweet)}), truncating.")
                # Try to truncate before URL
                if story_url and story_url in tweet:
                    parts = tweet.split(story_url)
                    text_part = parts[0].strip()[:180]  # Leave room for URL
                    tweet = f"{text_part}...\n\n{story_url}"
                else:
                    tweet = tweet[:277] + "..."

            # Final validation
            if not tweet or len(tweet) < 10:
                if story_url:
                    tweet = f"{topic}\n\n{story_url}\n\nThoughts?"
                else:
                    tweet = f"{topic} is making waves in tech. What's your take?"

            strategy["content"] = [tweet]
            strategy["source_url"] = story_url  # Track the source
            strategy["category"] = story.get('category', 'tech')  # Track category

        # FINAL VALIDATION: AI checks if everything makes sense before posting
        logger.info("Running final validation check on strategy...")
        validation_result = self._validate_strategy(strategy)

        if validation_result['valid']:
            logger.info(f"✓ Strategy validated: {validation_result['reason']}")
            return strategy
        else:
            logger.warning(f"✗ Validation rejected: {validation_result['reason']}")
            # On final attempt, be more lenient
            if attempt >= 2:
                logger.warning("Final attempt - accepting despite validation concerns")
                return strategy
            else:
                raise ValueError(f"Strategy validation failed: {validation_result['reason']}")

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
