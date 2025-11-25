import logging
import vertexai
from typing import List, Optional, Dict
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview.generative_models import grounding
from vertexai.preview.vision_models import ImageGenerationModel
from google.cloud import firestore
from config import Config
from news_fetcher import NewsFetcher
from tone_validator import ToneValidator
import datetime
import os
import re
from tenacity import retry, stop_after_attempt, wait_exponential

# Optional imports for new features
try:
    from youtube_fetcher import YouTubeFetcher
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False
    YouTubeFetcher = None

try:
    from infographic_generator import InfographicGenerator
    INFOGRAPHIC_AVAILABLE = True
except ImportError:
    INFOGRAPHIC_AVAILABLE = False
    InfographicGenerator = None

try:
    from scheduler import HumanScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    HumanScheduler = None

try:
    from content_mixer import ContentMixer
    MIXER_AVAILABLE = True
except ImportError:
    MIXER_AVAILABLE = False
    ContentMixer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentBrain:
    def __init__(self):
        self.project_id = Config.PROJECT_ID
        self.location = Config.REGION
        
        vertexai.init(project=self.project_id, location=self.location)
        aiplatform.init(project=self.project_id, location=self.location)

        self.model_names = []
        self.models = {}

        # Dynamic model discovery from Vertex AI
        candidate_models = self._discover_available_models()

        if not candidate_models:
            raise RuntimeError("No Gemini models discovered. Check Vertex AI API access.")

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

        # Initialize tone validator with dynamic pattern matching
        self.tone_validator = ToneValidator()
        logger.info("✓ Tone validator initialized with pattern-based validation")

        # Initialize YouTube fetcher for infographic topics
        self.youtube_fetcher = None
        if YOUTUBE_AVAILABLE:
            try:
                self.youtube_fetcher = YouTubeFetcher()
                logger.info("✓ YouTube fetcher initialized for infographic topics")
            except Exception as e:
                logger.warning(f"YouTube fetcher not available: {e}")

        # Initialize infographic generator
        self.infographic_generator = None
        if INFOGRAPHIC_AVAILABLE:
            try:
                self.infographic_generator = InfographicGenerator()
                logger.info("✓ Infographic generator initialized (Imagen 3)")
            except Exception as e:
                logger.warning(f"Infographic generator not available: {e}")

        # Initialize human-like scheduler
        self.scheduler = None
        if SCHEDULER_AVAILABLE:
            try:
                self.scheduler = HumanScheduler()
                logger.info("✓ Human scheduler initialized (Australia/Sydney)")
            except Exception as e:
                logger.warning(f"Scheduler not available: {e}")

        # Initialize content mixer for varied post types
        self.content_mixer = None
        if MIXER_AVAILABLE:
            try:
                self.content_mixer = ContentMixer(
                    news_fetcher=self.news_fetcher,
                    youtube_fetcher=self.youtube_fetcher,
                    infographic_generator=self.infographic_generator,
                    scheduler=self.scheduler
                )
                logger.info("✓ Content mixer initialized for varied posting")
            except Exception as e:
                logger.warning(f"Content mixer not available: {e}")

    def _discover_available_models(self) -> list:
        """
        Dynamically discovers available Gemini models from Vertex AI.
        Queries the API and validates each model works before adding to pool.
        Returns list of working model names.
        """
        import requests
        from google.auth import default
        from google.auth.transport.requests import Request

        discovered_models = []

        # Step 1: Try to query Vertex AI Model Garden API for available models
        logger.info("Discovering available Gemini models from Vertex AI...")

        try:
            # Get credentials for API call
            credentials, project = default()
            credentials.refresh(Request())
            access_token = credentials.token

            # Query the publisher models endpoint
            api_url = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models"

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            response = requests.get(api_url, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                models = data.get("models", data.get("publisherModels", []))

                for model in models:
                    model_name = model.get("name", "").split("/")[-1]
                    # Filter for Gemini models only
                    if "gemini" in model_name.lower():
                        discovered_models.append(model_name)
                        logger.info(f"  API discovered: {model_name}")

                logger.info(f"API returned {len(discovered_models)} Gemini models")
            else:
                logger.warning(f"Model API returned {response.status_code}: {response.text[:200]}")

        except Exception as e:
            logger.warning(f"Could not query Model Garden API: {e}")

        # Step 2: If API didn't return models, build list from known patterns
        if not discovered_models:
            logger.info("Building model list from known Vertex AI patterns...")
            # These patterns are based on Vertex AI naming conventions
            # Format: gemini-{version}-{variant}-{release}
            base_patterns = [
                # Gemini 2.0 series (latest)
                "gemini-2.0-flash-001",
                "gemini-2.0-flash-exp",
                "gemini-2.0-pro-exp",
                # Gemini 1.5 series (stable)
                "gemini-1.5-flash-002",
                "gemini-1.5-flash-001",
                "gemini-1.5-pro-002",
                "gemini-1.5-pro-001",
                # Experimental variants
                "gemini-1.5-flash-latest",
                "gemini-1.5-pro-latest",
                "gemini-exp-1206",
            ]
            discovered_models = base_patterns

        # Step 3: Validate each model actually works
        logger.info(f"Validating {len(discovered_models)} candidate models...")
        working_models = []

        for model_name in discovered_models:
            try:
                model = GenerativeModel(model_name)
                # Minimal test - just check model loads and responds
                test_response = model.generate_content(
                    "Say OK",
                    generation_config={"max_output_tokens": 5, "temperature": 0}
                )

                if test_response and test_response.text:
                    self.models[model_name] = model
                    self.model_names.append(model_name)
                    working_models.append(model_name)
                    logger.info(f"  ✓ {model_name} - working")
                else:
                    logger.debug(f"  ✗ {model_name} - empty response")

            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg or "not found" in error_msg.lower():
                    logger.debug(f"  ✗ {model_name} - not available")
                elif "429" in error_msg or "quota" in error_msg.lower():
                    # Quota error means model exists but we hit limits - still add it
                    logger.warning(f"  ⚠ {model_name} - quota limited, adding anyway")
                    self.models[model_name] = GenerativeModel(model_name)
                    self.model_names.append(model_name)
                    working_models.append(model_name)
                else:
                    logger.warning(f"  ✗ {model_name} - error: {error_msg[:80]}")

        # Step 4: Sort by preference (flash models first for speed/cost)
        def model_priority(name):
            if "2.0" in name and "flash" in name:
                return 0  # Prefer 2.0 flash
            elif "1.5" in name and "flash" in name:
                return 1  # Then 1.5 flash
            elif "2.0" in name:
                return 2  # Then 2.0 pro
            elif "1.5" in name and "pro" in name:
                return 3  # Then 1.5 pro
            else:
                return 4  # Others last

        self.model_names.sort(key=model_priority)
        logger.info(f"Model priority order: {self.model_names}")

        return working_models

    def should_post_now(self) -> tuple:
        """
        Uses scheduler to determine if we should post now.
        Returns (should_post: bool, reason: str).
        """
        if self.scheduler:
            return self.scheduler.should_post_now()
        return True, "No scheduler configured, always post"

    def get_preferred_post_types(self) -> list:
        """
        Gets time-appropriate post types from scheduler.
        """
        if self.scheduler:
            return self.scheduler.get_preferred_post_types()
        return ['text', 'image', 'video', 'meme', 'infographic']

    def generate_infographic(self, topic: str, key_points: list = None, source_url: str = None) -> dict:
        """
        Generates an infographic image for educational content.
        Returns dict with 'image_path', 'caption', 'image_prompt'.
        """
        if not self.infographic_generator:
            raise ValueError("Infographic generator not available")

        logger.info(f"Generating infographic for: {topic}")

        # Select style based on topic
        style = self.infographic_generator.select_style(topic)

        # Generate the infographic
        image_path = self.infographic_generator.generate(
            topic=topic,
            key_points=key_points,
            style=style
        )

        # Generate caption
        caption = self._generate_infographic_caption(topic, key_points, source_url)

        # Get the prompt used (for logging)
        image_prompt = self.infographic_generator.generate_infographic_prompt(
            topic=topic,
            key_points=key_points,
            style=style
        )

        return {
            'image_path': image_path,
            'content': caption,
            'image_prompt': image_prompt,
            'style': style
        }

    def _generate_infographic_caption(self, topic: str, key_points: list = None, source_url: str = None) -> str:
        """Generates a caption for infographic post."""
        # Use AI to generate engaging caption
        points_text = ""
        if key_points:
            points_text = f"Key concepts: {', '.join(key_points[:3])}"

        caption_prompt = f"""Write a brief, engaging caption for an infographic about:

Topic: {topic}
{points_text}

Requirements:
- 80-150 characters
- Informative but casual tone
- No hashtags, no emojis
- Sound like a human sharing educational content
- End with punctuation

Good examples:
- "Breaking down how transformers actually work. The attention mechanism visualized."
- "AI model sizes compared. From GPT-2 to GPT-4, the scale is wild."
- "Bitcoin mining explained in one graphic. Proof-of-work demystified."

Write the caption:"""

        try:
            caption = self._generate_with_fallback(caption_prompt)
            caption = caption.strip().strip('"')

            # Add source URL if provided
            if source_url:
                if len(caption) + len(source_url) + 4 <= 280:
                    caption = f"{caption}\n\n{source_url}"
                else:
                    max_len = 280 - len(source_url) - 7
                    caption = f"{caption[:max_len]}...\n\n{source_url}"

            return caption

        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")
            base = f"Explaining: {topic[:100]}"
            if source_url:
                return f"{base}\n\n{source_url}"
            return base

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
        Includes retry logic for transient errors (429, 503, etc.)

        If require_url=True and tools are provided, validates that response contains real URLs.
        """
        import time
        last_error = None
        max_retries_per_model = 2
        transient_error_codes = ['429', '503', '500', 'quota', 'rate', 'overloaded']

        for model_name in self.model_names:
            if model_name not in self.models:
                continue

            for retry in range(max_retries_per_model):
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
                                break  # Try next model, not retry

                            logger.info(f"✓ Generated content with {model_name} (found {len(valid_urls)} valid URLs)")
                        else:
                            logger.info(f"✓ Generated content with {model_name}")

                        return text
                    else:
                        logger.warning(f"✗ {model_name} returned empty response")
                        break  # Try next model

                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()

                    # Check if it's a transient error worth retrying
                    is_transient = any(code in error_str for code in transient_error_codes)

                    if is_transient and retry < max_retries_per_model - 1:
                        wait_time = (retry + 1) * 2  # Exponential backoff: 2s, 4s
                        logger.warning(f"⚠ {model_name} transient error, retrying in {wait_time}s... ({str(e)[:60]})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"✗ {model_name} failed: {str(e)[:100]}")
                        break  # Try next model

        # All models failed
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def _extract_article_context(self, title: str, url: str, html_content: str, category: str = "tech") -> str:
        """
        Extracts key context from article content using AI.
        Returns a concise summary of key points for media generation.
        Category determines the audience focus (ai/crypto/finance/tech).
        """
        if not html_content or len(html_content) < 100:
            logger.warning("No article content available, using title only")
            return f"Article: {title}"

        # Determine audience based on category
        audience_map = {
            "ai": "AI engineers and ML practitioners",
            "crypto": "crypto traders and blockchain enthusiasts",
            "finance": "tech investors and financial analysts",
            "tech": "tech enthusiasts and early adopters"
        }
        target_audience = audience_map.get(category, "tech community")

        # Ask AI to extract key details with category-specific angle
        context_prompt = f"""Extract the KEY DETAILS from this article for creating engaging media content.

ARTICLE TITLE: {title}
ARTICLE CONTENT (HTML):
{html_content[:8000]}

TARGET AUDIENCE: {target_audience}

Extract and return ONLY:
1. Main announcement/news (1 sentence)
2. Key details or specs (2-3 bullet points with NUMBERS/METRICS)
3. Why {target_audience} should care (1 sentence)

Be SPECIFIC - include numbers, names, metrics, prices, percentages from the article.
Keep total output under 400 characters.

Format:
NEWS: [main point]
DETAILS:
- [specific detail with numbers]
- [specific detail with metrics]
- [specific detail with data]
WHY: [impact/relevance to {target_audience}]
"""

        try:
            context = self._generate_with_fallback(context_prompt)
            logger.info(f"Extracted article context ({category}): {context[:100]}...")
            return context
        except Exception as e:
            logger.warning(f"Failed to extract context: {e}, using title only")
            return f"Article: {title}"

    def _get_trending_story(self, preferred_categories: List[str] = None) -> dict:
        """
        Gets a trending tech story with REAL URL from Hacker News or other sources.
        Fetches article content for rich context.
        Returns dict with {title, url, source, context, category}.
        """
        story = self.news_fetcher.get_trending_story(preferred_categories=preferred_categories)

        if story:
            logger.info(f"✓ Found trending story: {story['title'][:50]}...")

            # Fetch article content for context with category-specific audience
            if story.get('url'):
                html_content = self.news_fetcher.fetch_article_content(story['url'])
                category = story.get('category', 'tech')  # Get category from story or default to 'tech'
                context = self._extract_article_context(story['title'], story['url'], html_content, category)
                story['context'] = context
            else:
                story['context'] = f"Article: {story['title']}"

            return story

        # Fallback: use model to suggest a topic (no URL)
        logger.warning("Could not fetch real news, falling back to model knowledge")
        prompt = """Suggest ONE specific, real tech product or project that tech enthusiasts would find interesting.
        Examples: "Next.js 15", "Anthropic Claude 3.5 Sonnet", "Meta Llama 3"

        Return ONLY the name. Be specific and real."""

        topic_name = self._generate_with_fallback(prompt)
        return {
            'title': topic_name,
            'url': None,  # No URL available
            'source': 'model_knowledge',
            'category': 'tech',
            'context': f"Topic: {topic_name}"
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _check_history(self, topic: str, url: str = None) -> bool:
        """
        Checks Firestore to see if we've recently posted about this topic or URL.
        Returns True if we should SKIP this topic (duplicate), False otherwise.
        """
        try:
            # Check last 30 posts (increased from 20 for better duplicate detection)
            docs = self.collection.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(30).stream()

            def _normalize(text):
                return set(text.lower().split())

            current_words = _normalize(topic)

            for doc in docs:
                data = doc.to_dict()
                stored_topic = data.get("topic", "")
                stored_url = data.get("source_url", "")

                # PRIORITY CHECK: Same URL = definitely a duplicate
                if url and stored_url and url == stored_url:
                    logger.info(f"URL '{url}' was already posted. Skipping to avoid duplicate.")
                    return True

                # Keyword overlap check (Jaccard similarity) for topic
                stored_words = _normalize(stored_topic)
                if not current_words: continue
                overlap = len(stored_words & current_words) / len(current_words)

                if overlap > 0.6: # 60% overlap
                    logger.info(f"Topic '{topic}' matches recent post '{stored_topic}' (Overlap: {overlap:.2f}). Skipping.")
                    return True
            return False
        except Exception as e:
            logger.warning(f"Firestore history check failed: {e}. Proceeding without check.")
            return False

    def _get_recent_categories(self, limit: int = 10) -> List[str]:
        """
        Gets categories of recent posts for variety and balance tracking.
        Returns list of category names from most recent to oldest.
        """
        try:
            docs = self.collection.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit).stream()
            categories = []
            for doc in docs:
                data = doc.to_dict()
                category = data.get("category", "tech")
                categories.append(category)

            logger.info(f"Recent {len(categories)} post categories: {categories}")
            return categories
        except Exception as e:
            logger.warning(f"Failed to get recent categories: {e}")
            return []

    def _get_preferred_categories(self) -> List[str]:
        """
        Determines preferred categories based on recent post history.
        Enforces variety (no 3+ same in a row) and balance (diversify if one dominates).
        """
        recent_10 = self._get_recent_categories(10)
        recent_3 = recent_10[:3] if len(recent_10) >= 3 else recent_10

        # VARIETY CHECK: Last 3 posts same category?
        if len(recent_3) >= 3 and len(set(recent_3)) == 1:
            repeated_category = recent_3[0]
            logger.warning(f"Last 3 posts all {repeated_category}. Forcing variety!")
            # Return all OTHER categories
            all_categories = ['ai', 'crypto', 'finance', 'tech']
            preferred = [cat for cat in all_categories if cat != repeated_category]
            logger.info(f"Variety enforcement: preferred categories = {preferred}")
            return preferred

        # BALANCE CHECK: Count category distribution in last 10
        if len(recent_10) >= 5:
            from collections import Counter
            counts = Counter(recent_10)
            total = len(recent_10)

            # Find if any category is over-represented (>50%)
            for category, count in counts.most_common():
                percentage = (count / total) * 100
                if percentage > 50:
                    logger.warning(f"Category '{category}' is {percentage:.0f}% of last {total} posts. Rebalancing!")
                    # Prefer categories that are under-represented
                    all_categories = ['ai', 'crypto', 'finance', 'tech']
                    # Sort by count (ascending) - prefer least used
                    preferred = sorted(all_categories, key=lambda cat: counts.get(cat, 0))
                    logger.info(f"Balance enforcement: preferred categories = {preferred}")
                    return preferred

        # Default: normal priority (AI > crypto > finance > tech)
        return ['ai', 'crypto', 'finance', 'tech']

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

        # Dynamic tone validation using pattern-based validator
        is_valid_tone, tone_issue = self.tone_validator.validate(content_text)
        if not is_valid_tone:
            return {
                'valid': False,
                'reason': tone_issue
            }

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

4. ✓ Is tone CASUAL and HUMAN (not formal/robotic/preachy)?
   - REJECT preachy: "Good to see...", "Nice to see...", "Great to see..."
   - REJECT meta-commentary: "wild to think about", "makes sense", "worth noting"
   - REJECT filler endings: "honestly.", "frankly.", "finally.", "really."
   - REJECT filler starts: "So,", "Well,", "Look,"
   - REJECT hedging: "starting to", "seems like", "feels like"
   - REJECT forced casual: "gotta", "wanna", "gonna" (unless natural)
   - REJECT formal questions: "How will this impact", "What does this mean for"
   - APPROVE punchy, direct: "About time.", "Progress.", "This matters."
   - Sound like a real person, not observing from outside

EXAMPLES OF BAD TONE TO REJECT:
{chr(10).join(self.tone_validator.get_bad_examples())}
- "Unleash your creative vision" - Marketing language
- "Check back later for updates!" - Placeholder text
- Mentions products not in the original topic - Made up

EXAMPLES OF GOOD TONE TO APPROVE:
{chr(10).join(self.tone_validator.get_good_examples())}

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
        # Get preferred categories based on recent post history (variety + balance)
        preferred_categories = self._get_preferred_categories()

        story = self._get_trending_story(preferred_categories=preferred_categories)
        topic = story['title']
        story_url = story.get('url')  # Real URL or None
        story_context = story.get('context', f"Article: {topic}")  # Rich context from article

        # Retry logic for duplicates (check both topic AND URL)
        if self._check_history(topic, story_url):
            logger.info("Duplicate detected (topic or URL). Requesting alternative.")
            # Try to get a different story
            try:
                for _ in range(3):  # Try up to 3 times
                    story = self._get_trending_story(preferred_categories=preferred_categories)
                    topic = story['title']
                    story_url = story.get('url')
                    story_context = story.get('context', f"Article: {topic}")

                    if not self._check_history(topic, story_url):
                        break
                else:
                    logger.warning("All alternatives were duplicates, proceeding with latest")
            except Exception as e:
                logger.error(f"Failed to find alternative topic: {e}")
                # Proceed with original topic if fallback fails, better than crashing

        logger.info(f"Selected Topic: {topic}")
        logger.info(f"Article Context: {story_context[:150]}...")

        # Decide format
        if Config.BUDGET_MODE:
            logger.info("BUDGET_MODE enabled, using text-only format")
            post_type = "text"
        else:
            logger.info("BUDGET_MODE disabled, deciding optimal format for media generation")
            # Ask AI if media actually adds value or if text + link is better
            decision_prompt = f"""For this tech news, decide the BEST format for maximum engagement.

ARTICLE CONTEXT:
{story_context}

TOPIC: {topic}
HAS URL: {'Yes - Twitter will show preview card' if story_url else 'No URL available'}

DECISION CRITERIA:

Choose VIDEO only if:
- Article explains a PROCESS or WORKFLOW (step-by-step)
- Article describes HOW something works (algorithm, architecture)
- Visual sequence would make it clearer
- Example: "How transformer attention works", "Video generation pipeline"

Choose IMAGE only if:
- Article is about a NEW PRODUCT/DEVICE that needs visualization
- Complex architecture/diagram would help understanding
- Before/after comparison is meaningful
- Example: "New chip design", "UI redesign comparison"

Choose INFOGRAPHIC if:
- Article explains a CONCEPT that would benefit from educational visualization
- Complex topic that can be broken down into visual components
- Statistics, comparisons, or trends that work well as data viz
- Topic is educational and could be shown as a diagram/chart
- Example: "AI model comparison", "Evolution of programming languages", "Tech market trends"

Choose MEME if:
- Story is ironic, contradictory, or absurd (perfect for meme format)
- Situation is relatable and funny to tech community
- Can be expressed as reaction/commentary meme
- Example: "Another AI company claiming AGI", "Tech layoffs then hiring spree", "New JS framework drops"

Choose TEXT if:
- Simple announcement or partnership (like "X partners with Y")
- Financial/business news without technical process
- Twitter link preview card is sufficient
- Media would be redundant with preview
- Example: "Company raises $X", "Partnership announced"

IMPORTANT: Don't generate redundant media just for engagement. If the Twitter link preview card shows the story well, use TEXT. MEME should be used sparingly (every 5-7 posts). INFOGRAPHIC is great for educational content.

Reply with EXACTLY ONE WORD: VIDEO, IMAGE, INFOGRAPHIC, MEME, or TEXT"""

            try:
                decision = self._generate_with_fallback(decision_prompt).upper()
                logger.info(f"Format decision response: {decision}")

                if "VIDEO" in decision:
                    post_type = "video"
                elif "INFOGRAPHIC" in decision:
                    post_type = "infographic"
                elif "MEME" in decision:
                    post_type = "meme"
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
            # Generate Video Prompt and Tweet Text with FULL article context
            # Determine if this needs an explainer video or a hook video
            script_prompt = f"""Generate a tweet with video for THIS EXACT NEWS STORY.

ARTICLE CONTEXT (READ CAREFULLY - USE THIS INFO):
{story_context}

ARTICLE TITLE: {topic}
SOURCE URL: {story_url if story_url else 'No URL'}

CRITICAL WARNING: You MUST write about THIS EXACT article using the context above. DO NOT make up fake products or features!

VIDEO TYPE DECISION:
- If article involves a process, algorithm, or how something works → EXPLAINER video
- If article is breaking news, announcement, or debate → HOOK video

You MUST provide BOTH parts in this EXACT format:
CAPTION: <your complete tweet text here>
PROMPT: <your detailed visual description here>

CAPTION REQUIREMENTS:
- Must reference the ACTUAL article content from context above
- Use SPECIFIC details from the article context (numbers, names, features)
- Do NOT invent product names, versions, or features beyond what's in the context
- Must be a COMPLETE sentence ending with punctuation (. ! ?)
- 100-175 characters total (leaves room for URL)
- Sound CASUAL and HUMAN, not robotic or formal
- NO formal questions like "How will this impact..." or "What does this mean for..."
- NO marketing language ("Unleash", "Revolutionary", etc.)
- NO hashtags, NO emojis

PROMPT REQUIREMENTS FOR VIDEO (IMPORTANT - BE DETAILED AND SPECIFIC TO THE ARTICLE):

For EXPLAINER videos (processes, how-to, algorithms):
- Describe a clear visual sequence showing the SPECIFIC process from the article
- Include diagrams, flowcharts, or code related to the ACTUAL technology mentioned
- Show before/after states or transformations SPECIFIC to the article
- Use actual numbers/metrics from the article context
- Example: "Animated flowchart showing data moving through neural network layers, with nodes lighting up sequentially as computation progresses, ending with output prediction appearing"

For HOOK videos (news, announcements, debates):
- Start with attention-grabbing visual related to the ACTUAL company/product in the article
- Include relevant tech imagery specific to what's described in the context
- Create visual intrigue based on the REAL story details
- Use actual logos, products, or visuals mentioned in the context
- Example: "Zoom into glowing AI chip with circuit patterns, transition to split-screen comparison of old vs new performance graphs showing 2x speedup, end on provocative question mark"

PROMPT SHOULD BE:
- 100-200 characters (detailed and specific to THIS article)
- Cinematically interesting but FACTUAL to the article
- Technically relevant using details from the context
- Actually achievable by a video generator
- References REAL specs, numbers, or details from the article

BAD Examples (NEVER DO THIS):
X "Unleash creativity with Nano Banana Pro!" (made up product)
X "Check back later for the video!" (placeholder)
X "Tech-focused, developer-oriented visuals" (too generic, not specific to article)
X "Cool AI stuff" (not specific enough, ignores article context)

GOOD Examples (USING ARTICLE CONTEXT):
If article says "GPT-5 reduces hallucinations by 40%":
CAPTION: "GPT-5 cuts hallucinations by 40%. Finally getting somewhere with reliability."
PROMPT: "Split screen showing GPT-4 vs GPT-5 accuracy charts, bars rising to show 40% improvement, transition to checkmark appearing over error-prone outputs"

If article says "Rust adoption grows 67% among Fortune 500":
CAPTION: "Fortune 500 went 67% more Rust this year. Memory safety wins."
PROMPT: "Animated bar chart racing showing programming language adoption, Rust bar surging 67% upward past other languages, corporate logos appearing on rising bar"

BAD (too formal):
X "How will this impact the future of AI reliability?" (robotic question)
X "What does increased Rust adoption mean for enterprise?" (textbook tone)

CRITICAL: Write COMPLETE, STANDALONE caption using REAL details from the article context. NO placeholder text!

Now generate for the article above.
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

                    # Reserve space for URL if needed (Twitter limit 280 chars)
                    # Estimate URL space: typical URL ~100 chars + "\n\n" = 104 chars
                    # So caption should be max 175 chars to leave room
                    max_caption_len = 175 if story_url else 280
                    caption = caption_part[:max_caption_len]

                    # Validate visual prompt is detailed enough
                    if len(visual_prompt) < 50:
                        logger.warning(f"Video prompt too short ({len(visual_prompt)} chars): {visual_prompt}")
                        raise ValueError(f"Video prompt must be at least 50 characters, got {len(visual_prompt)}")

                    logger.info(f"Video prompt: {visual_prompt[:100]}...")
                else:
                    logger.error("Response missing CAPTION: or PROMPT: markers")
                    raise ValueError("Invalid format - missing CAPTION or PROMPT")

            except Exception as e:
                logger.error(f"Failed to generate video script: {e}")
                raise

            # For video posts with URL, add URL to caption for citation
            if story_url:
                if story_url not in caption:
                    # Add URL to caption (Twitter limit is 280 chars)
                    if len(caption) + len(story_url) + 4 <= 280:  # 4 for "\n\n" and buffer
                        caption = f"{caption}\n\n{story_url}"
                    else:
                        # Truncate caption to fit URL within 280 char limit
                        max_cap_len = 280 - len(story_url) - 7  # 7 for "...\n\n"
                        caption = f"{caption[:max_cap_len]}...\n\n{story_url}"
                    logger.info(f"Added URL to video caption: {story_url}")

            strategy["content"] = caption
            strategy["video_prompt"] = visual_prompt

        elif post_type == "image":
            # Generate Image Prompt and Tweet Text with FULL article context
            script_prompt = f"""Generate a tweet with image for THIS EXACT NEWS STORY.

ARTICLE CONTEXT (READ CAREFULLY - USE THIS INFO):
{story_context}

ARTICLE TITLE: {topic}
SOURCE URL: {story_url if story_url else 'No URL'}

CRITICAL WARNING: You MUST write about THIS EXACT article using the context above. DO NOT make up fake products or features!

You MUST provide BOTH parts in this EXACT format:
CAPTION: <your complete tweet text here>
PROMPT: <your detailed visual description here>

CAPTION REQUIREMENTS:
- Must reference the ACTUAL article content from context above
- Use SPECIFIC details from the article context (numbers, names, features)
- Do NOT invent product names, versions, or features beyond what's in the context
- Must be a COMPLETE sentence ending with punctuation (. ! ?)
- 100-175 characters total (leaves room for URL)
- Sound CASUAL and HUMAN, not robotic or formal
- NO formal questions like "How will this impact..." or "What does this mean for..."
- NO marketing language ("Unleash", "Revolutionary", etc.)
- NO hashtags, NO emojis

PROMPT REQUIREMENTS FOR IMAGE (IMPORTANT - BE SPECIFIC TO THE ARTICLE):
- Visual description for image generator (Imagen) based on ACTUAL article content
- Must DIRECTLY represent the SPECIFIC feature/product/technology mentioned in article
- Use tech photography or technical illustration style
- Include specific visual elements that MATCH the article's key feature/concept
- 80-150 characters with concrete details
- Reference actual products, logos, interfaces, interactions from the article
- CRITICAL: Image must visually show what the article is about, not generic tech imagery

EXAMPLES USING ARTICLE CONTEXT:
If article says "Gemini lets you tap on image parts for definitions":
CAPTION: "Gemini lets you tap on image parts for instant definitions. Finally, a feature that actually makes sense."
PROMPT: "Smartphone screen showing Gemini app with an educational image, finger tapping on a specific object, definition popup appearing with explanation text, interactive UI elements glowing"

If article says "New MacBook Pro M4 chip benchmarks leaked":
CAPTION: "M4 MacBook Pro benchmarks leaked. 30% faster than M3. Expensive upgrade season incoming."
PROMPT: "Professional product photo of MacBook Pro with glowing M4 chip visualization, performance graphs floating above screen showing 30% increase, dramatic tech lighting"

If article says "GitHub Copilot now writes entire functions":
CAPTION: "Copilot now writes entire functions. Junior dev job market about to get interesting."
PROMPT: "Split screen showing code editor with GitHub Copilot logo, left side showing developer typing partial code, right side auto-completing entire function, sleek modern tech aesthetic"

BAD (too formal):
X "How can this deepen learning experiences?" (robotic)
X "Are junior developers obsolete?" (textbook question)

BAD Examples (NEVER DO THIS):
X "Unleash creativity with Gemini 3 Pro Image!" (made up, marketing)
X "Check back later for updates!" (placeholder)
X "Tech photography" (too generic, not specific to article)
X Any placeholder or filler text

CRITICAL: Write COMPLETE, STANDALONE caption using REAL details from the article context. NO placeholder text!

Now generate for the article above.
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

                    # Reserve space for URL if needed (Twitter limit 280 chars)
                    # Estimate URL space: typical URL ~100 chars + "\n\n" = 104 chars
                    # So caption should be max 175 chars to leave room
                    max_caption_len = 175 if story_url else 280
                    caption = caption_part[:max_caption_len]

                    # Validate image prompt is detailed enough and specific
                    if len(visual_prompt) < 50:
                        logger.warning(f"Image prompt too short ({len(visual_prompt)} chars): {visual_prompt}")
                        raise ValueError(f"Image prompt must be at least 50 characters, got {len(visual_prompt)}")

                    logger.info(f"Image prompt: {visual_prompt[:100]}...")
                else:
                    logger.error("Response missing CAPTION: or PROMPT: markers")
                    raise ValueError("Invalid format - missing CAPTION or PROMPT")

            except Exception as e:
                logger.error(f"Failed to generate image script: {e}")
                raise

            # For image posts with URL, add URL to caption for citation
            if story_url:
                if story_url not in caption:
                    # Add URL to caption (Twitter limit is 280 chars)
                    if len(caption) + len(story_url) + 4 <= 280:  # 4 for "\n\n" and buffer
                        caption = f"{caption}\n\n{story_url}"
                    else:
                        # Truncate caption to fit URL within 280 char limit
                        max_cap_len = 280 - len(story_url) - 7  # 7 for "...\n\n"
                        caption = f"{caption[:max_cap_len]}...\n\n{story_url}"
                    logger.info(f"Added URL to image caption: {story_url}")

            strategy["content"] = caption
            strategy["image_prompt"] = visual_prompt

        elif post_type == "meme":
            # Generate Meme about current tech affairs
            logger.info(f"Generating meme for: {topic}")

            meme_prompt = f"""Generate a MEME about this tech news story.

ARTICLE CONTEXT:
{story_context}

TOPIC: {topic}

You MUST provide BOTH parts in this EXACT format:
CAPTION: <your meme caption text here>
PROMPT: <detailed meme image description for Imagen>

MEME CAPTION REQUIREMENTS:
- Short, punchy, meme-style text (50-150 chars)
- Sound CASUAL and RELATABLE to tech community
- Can be sarcastic, ironic, or observational
- Reference the ACTUAL story with humor
- NO formal language, NO marketing speak
- Global perspective (bot is Australian)

MEME IMAGE PROMPT REQUIREMENTS:
- Describe a REACTION IMAGE or MEME FORMAT
- Be specific about expression/emotion
- Meme-worthy situation or comparison
- Classic meme styles work: "Drake approving/disapproving", "Distracted boyfriend", "This is fine", etc.
- Or describe reaction: "Person looking shocked", "Side-eye glance", "Facepalm"
- 80-150 characters

GOOD MEME EXAMPLES:
Topic: "Another startup claims AGI breakthrough"
CAPTION: "Another AGI announcement. Sure mate, right after Duke Nukem Forever ships."
PROMPT: "Side-eye meme format, person giving suspicious skeptical look to camera, doubtful expression"

Topic: "Meta lays off AI team then posts 50 AI job openings"
CAPTION: "Meta: Fires AI team. Also Meta: Now hiring AI engineers. Make it make sense."
PROMPT: "Two button meme format, person sweating choosing between two contradictory buttons, corporate confusion"

Topic: "New JavaScript framework promises to end framework fatigue"
CAPTION: "New JS framework to end framework fatigue. The irony is not lost on us."
PROMPT: "This is fine meme, person sitting in burning room drinking coffee, resigned acceptance"

Now generate the meme:
"""

            try:
                response = self._generate_with_fallback(meme_prompt)
                logger.info(f"Meme generation response: {response[:100]}...")

                if "CAPTION:" in response and "PROMPT:" in response:
                    parts = response.split("PROMPT:")
                    caption_part = parts[0].replace("CAPTION:", "").strip()
                    meme_image_prompt = parts[1].strip()

                    # Validate caption
                    if len(caption_part) < 20 or not any(caption_part.endswith(p) for p in ['.', '!', '?']):
                        logger.warning(f"Meme caption seems incomplete: {caption_part}")
                        caption_part = f"{caption_part}."

                    max_caption_len = 175 if story_url else 280
                    caption = caption_part[:max_caption_len]

                    # Validate meme prompt
                    if len(meme_image_prompt) < 30:
                        logger.warning(f"Meme prompt too short: {meme_image_prompt}")
                        raise ValueError(f"Meme prompt must be detailed (30+ chars)")

                    logger.info(f"Meme image prompt: {meme_image_prompt[:100]}...")
                else:
                    logger.error("Response missing CAPTION: or PROMPT: markers")
                    raise ValueError("Invalid format - missing CAPTION or PROMPT")

            except Exception as e:
                logger.error(f"Failed to generate meme: {e}")
                raise

            # Add URL to caption if available
            if story_url:
                if story_url not in caption:
                    if len(caption) + len(story_url) + 4 <= 280:
                        caption = f"{caption}\n\n{story_url}"
                    else:
                        max_cap_len = 280 - len(story_url) - 7
                        caption = f"{caption[:max_cap_len]}...\n\n{story_url}"
                    logger.info(f"Added URL to meme caption: {story_url}")

            strategy["content"] = caption
            strategy["image_prompt"] = meme_image_prompt  # Use same image generation as IMAGE type

        elif post_type == "infographic":
            # Generate educational infographic
            logger.info(f"Generating infographic for: {topic}")

            # Extract key points from article context for infographic
            key_points_prompt = f"""Extract 3-5 KEY CONCEPTS from this article for an educational infographic.

ARTICLE CONTEXT:
{story_context}

TOPIC: {topic}

Return ONLY a comma-separated list of short concepts (2-4 words each).
Example: "Neural Networks, Attention Mechanism, Training Data, Model Size, Inference Speed"

Key concepts:"""

            key_points = []
            try:
                key_points_response = self._generate_with_fallback(key_points_prompt)
                key_points = [kp.strip() for kp in key_points_response.split(',')][:5]
                logger.info(f"Extracted key points: {key_points}")
            except Exception as e:
                logger.warning(f"Key points extraction failed: {e}")
                key_points = ['Technology', 'Innovation', 'Digital']

            # Generate infographic caption
            infographic_caption_prompt = f"""Write a caption for an educational infographic about:

TOPIC: {topic}
KEY CONCEPTS: {', '.join(key_points)}

Requirements:
- 80-150 characters
- Informative but casual tone
- Sounds like a human sharing educational content
- NO hashtags, NO emojis
- End with punctuation

Good examples:
- "Breaking down how transformers work. The attention mechanism visualized."
- "AI model sizes compared. From GPT-2 to GPT-4, the scale is wild."
- "Bitcoin mining explained. Proof-of-work in one graphic."

Caption:"""

            try:
                caption = self._generate_with_fallback(infographic_caption_prompt)
                caption = caption.strip().strip('"')

                # Validate caption
                if len(caption) < 20 or not any(caption.endswith(p) for p in ['.', '!', '?']):
                    caption = f"{caption}."

                max_caption_len = 175 if story_url else 280
                caption = caption[:max_caption_len]

            except Exception as e:
                logger.warning(f"Caption generation failed: {e}")
                caption = f"Explaining: {topic[:100]}"

            # Generate infographic prompt for Imagen
            infographic_visual_prompt = f"""Professional tech infographic about "{topic}".

Key concepts to visualize: {', '.join(key_points)}

Style: Clean, modern data visualization diagram
Color scheme: Professional blue and white with accent colors
Visual elements: Charts, diagrams, icons, connecting lines

Requirements:
- Educational and informative
- High contrast, readable at small sizes
- Minimalist tech aesthetic
- 16:9 aspect ratio for Twitter/X
- No text clutter, visual explanation"""

            # Add URL to caption if available
            if story_url:
                if len(caption) + len(story_url) + 4 <= 280:
                    caption = f"{caption}\n\n{story_url}"
                else:
                    max_cap_len = 280 - len(story_url) - 7
                    caption = f"{caption[:max_cap_len]}...\n\n{story_url}"
                logger.info(f"Added URL to infographic caption: {story_url}")

            strategy["content"] = caption
            strategy["image_prompt"] = infographic_visual_prompt
            strategy["key_points"] = key_points

        else:
            # Generate Hacker News Style Post with REAL URL
            logger.info(f"Generating HN-style post for: {topic}")

            if story_url:
                # We have a REAL URL from news fetcher!
                logger.info(f"Using real URL: {story_url}")
                post_prompt = f"""Write a casual, engaging tweet about this tech news.

ARTICLE CONTEXT:
{story_context}

Title: "{topic}"
URL: {story_url}

TONE: Sound HUMAN and CASUAL, not like a bot or corporate account. Be conversational.

IMPORTANT: Just write the tweet directly. DO NOT include style labels like "Style A:", "Style B:", etc.

Pick ONE of these approaches (but don't label it):

Approach 1 - Statement + Short reaction:
"[Bold statement]. [Short punchy reaction]."

Approach 2 - Fact + Skeptical take:
"[Interesting fact]. [Slightly skeptical comment]."

Approach 3 - Direct observation:
"[What's happening]. [Why it matters]."

CONSTRAINTS:
- Total: Under 280 chars (including URL)
- Use EXACT URL provided: {story_url}
- NO hashtags, NO emojis
- NO style labels ("Style A:", "**Style B:**", etc.)
- NO formal questions like "How will this impact..." or "What does this mean for..."
- NO "We" or "Check out" or "Read more"
- NO US-centric language ("home soil", "came home", "stateside", "domestic")
- Global perspective - don't assume US is "home"
- Just write the tweet directly, nothing else

GOOD Examples (SHORT, PUNCHY, CASUAL, GLOBAL):
{chr(10).join([ex[2:] for ex in self.tone_validator.get_good_examples()])}

BAD Examples (NEVER DO THIS):
{chr(10).join(self.tone_validator.get_bad_examples())}

Generate the tweet:
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
