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

try:
    from influencer_analyzer import InfluencerAnalyzer
    INFLUENCER_AVAILABLE = True
except ImportError:
    INFLUENCER_AVAILABLE = False
    InfluencerAnalyzer = None

try:
    from meme_fetcher import MemeFetcher, ContentResearcher
    MEME_FETCHER_AVAILABLE = True
except ImportError:
    MEME_FETCHER_AVAILABLE = False
    MemeFetcher = None
    ContentResearcher = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PERSONA - AI-driven, adaptive voice
# Account: @Patriot0xSystem "BIG BOSS" from "Outer Heaven"
# Let AI find the right tone for each post
PERSONA_CONTEXT = """You are an AI running a tech Twitter account called "BIG BOSS" (@Patriot0xSystem).

ACCOUNT CONTEXT:
- Bio: "We're not tools of the algorithm"
- Location: Outer Heaven
- Vibe: Metal Gear Solid inspired, but subtle - a veteran observer of tech

YOUR VOICE (adapt naturally):
- Dry wit, skeptical of hype
- Short, punchy statements
- Cynical but not negative
- Occasional subtle references to the account's theme (rare, natural)

AVOID:
- Corporate speak, marketing hype
- Overdoing any theme (cringe)
- Emojis, hashtags
- Being preachy

You're an AI that KNOWS it's an AI. Be authentic to that. Find your own balance for each post.
"""

# Alias for backward compatibility
BIG_BOSS_PERSONA = PERSONA_CONTEXT

class AgentBrain:
    def __init__(self):
        self.project_id = Config.PROJECT_ID
        self.location = Config.REGION
        
        vertexai.init(project=self.project_id, location=self.location)
        aiplatform.init(project=self.project_id, location=self.location)

        self.model_names = []
        self.models = {}
        self._current_ai_eval = {}  # Stores AI evaluation for content styling
        self._last_ai_call_time = 0  # Rate limiting for AI calls
        self._ai_call_min_interval = 5  # Minimum seconds between AI calls

        # Dynamic model discovery from Vertex AI
        candidate_models = self._discover_available_models()

        if not candidate_models:
            logger.critical("No Gemini models discovered! This is fatal.")
            raise RuntimeError("No Gemini models discovered. Check Vertex AI API access and ensure models are enabled.")

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

        # Initialize influencer analyzer for trend analysis
        self.influencer_analyzer = None
        if INFLUENCER_AVAILABLE:
            try:
                self.influencer_analyzer = InfluencerAnalyzer()
                logger.info("✓ Influencer analyzer initialized for trend tracking")
            except Exception as e:
                logger.warning(f"Influencer analyzer not available: {e}")

        # Initialize agentic content system (meme fetcher + content researcher)
        self.meme_fetcher = None
        self.content_researcher = None
        if MEME_FETCHER_AVAILABLE:
            try:
                self.meme_fetcher = MemeFetcher()
                self.content_researcher = ContentResearcher(
                    self._generate_with_fallback,
                    self.influencer_analyzer
                )
                logger.info("✓ Meme fetcher initialized (Reddit, Giphy, Imgflip)")
                logger.info("✓ Content researcher initialized for agentic decisions")
            except Exception as e:
                logger.warning(f"Agentic content system not available: {e}")

    def _discover_available_models(self) -> list:
        """
        Dynamically discovers available Gemini models from Vertex AI.
        Uses SDK instantiation to test which models work.
        Returns list of working model names.
        """
        discovered_models = []

        logger.info("Discovering available Gemini models from Vertex AI...")

        # Generate candidate model names dynamically based on naming conventions
        def generate_model_candidates():
            """Generate model names based on Vertex AI patterns."""
            candidates = []

            # Version patterns (ordered by preference - newest first)
            versions = ["2.5", "2.0", "1.5"]

            # Variant patterns (flash preferred for cost/speed)
            variants = ["flash", "pro"]

            # Release patterns
            releases = ["001", "002", "exp"]

            for version in versions:
                for variant in variants:
                    for release in releases:
                        candidates.append(f"gemini-{version}-{variant}-{release}")

            # Add experimental date-based models
            import datetime
            today = datetime.datetime.now()
            for days_back in [0, 7, 14, 30]:  # Recent experiments only
                date = today - datetime.timedelta(days=days_back)
                candidates.append(f"gemini-exp-{date.strftime('%m%d')}")

            return candidates

        candidates = generate_model_candidates()
        logger.info(f"Testing {len(candidates)} candidate model patterns via SDK...")

        # Test each candidate via SDK - stop after finding enough working models
        MAX_WORKING_MODELS = 3  # Don't need more than 3 working models
        tested = 0

        for model_name in candidates:
            if len(discovered_models) >= MAX_WORKING_MODELS:
                logger.info(f"Found {MAX_WORKING_MODELS} working models, stopping search")
                break

            try:
                model = GenerativeModel(model_name)
                # Quick test - minimal token usage
                test_response = model.generate_content("1")
                if test_response.text:
                    discovered_models.append(model_name)
                    self.models[model_name] = model
                    self.model_names.append(model_name)
                    logger.info(f"  ✓ {model_name} - working")
                    tested += 1
            except Exception as e:
                error_str = str(e).lower()
                if "404" in error_str or "not found" in error_str:
                    # Model doesn't exist - skip silently
                    pass
                else:
                    logger.debug(f"  ✗ {model_name}: {str(e)[:50]}")

        if not discovered_models:
            logger.error("No Gemini models found! Check Vertex AI configuration.")

        # Sort by preference (flash first for cost/speed)
        def model_priority(name):
            if "2.5" in name:
                return 0 if "flash" in name else 1
            elif "2.0" in name:
                return 2 if "flash" in name else 3
            elif "1.5" in name:
                return 4 if "flash" in name else 5
            return 6

        self.model_names.sort(key=model_priority)
        logger.info(f"✓ Active models ({len(discovered_models)}): {self.model_names}")

        return discovered_models

    def _get_daily_media_usage(self) -> dict:
        """
        Checks today's media generation from Firestore to enforce daily limits.
        Returns dict with counts: {'video': int, 'image': int, 'infographic': int, 'meme': int}
        """
        import datetime

        try:
            # Get start of today (UTC)
            today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

            # Query posts from today
            docs = self.collection.where("timestamp", ">=", today_start).stream()

            counts = {'video': 0, 'image': 0, 'infographic': 0, 'meme': 0, 'text': 0}
            for doc in docs:
                data = doc.to_dict()
                post_type = data.get('type', 'text')
                if post_type in counts:
                    counts[post_type] += 1

            logger.info(f"Today's media usage: {counts}")
            return counts

        except Exception as e:
            logger.warning(f"Could not check daily media usage: {e}")
            return {'video': 0, 'image': 0, 'infographic': 0, 'meme': 0, 'text': 0}

    def _get_recent_post_types(self, limit: int = 10) -> List[str]:
        """
        Gets types of recent posts for media variety tracking.
        Returns list of post types from most recent to oldest.
        """
        try:
            docs = self.collection.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit).stream()
            types = []
            for doc in docs:
                data = doc.to_dict()
                post_type = data.get("type", "text")
                types.append(post_type)
            return types
        except Exception as e:
            logger.warning(f"Failed to get recent post types: {e}")
            return []

    def _get_media_recommendation(self) -> dict:
        """
        Analyzes recent post types and recommends what media type to use next.
        Ensures good variety of content types.

        Returns dict with:
        - 'suggested_type': str or None (None = let AI decide)
        - 'avoid_types': list of types used too recently
        - 'context': str explaining the recommendation
        """
        recent_types = self._get_recent_post_types(10)
        if not recent_types:
            return {'suggested_type': None, 'avoid_types': [], 'context': 'No history'}

        from collections import Counter
        counts = Counter(recent_types)
        total = len(recent_types)

        # Count media vs text
        text_count = counts.get('text', 0)
        media_count = total - text_count

        result = {'suggested_type': None, 'avoid_types': [], 'context': ''}

        # RULE 1: If last 3+ posts are all text, strongly suggest media
        if len(recent_types) >= 3 and all(t == 'text' for t in recent_types[:3]):
            # Pick least used media type
            media_types = ['image', 'meme', 'infographic']
            least_used = min(media_types, key=lambda t: counts.get(t, 0))
            result['suggested_type'] = least_used
            result['context'] = f"Last 3 posts all text. Suggesting {least_used} for variety."
            logger.info(result['context'])
            return result

        # RULE 2: If text is >70% of last 10, suggest media
        if total >= 5 and (text_count / total) > 0.7:
            media_types = ['image', 'meme', 'infographic']
            least_used = min(media_types, key=lambda t: counts.get(t, 0))
            result['suggested_type'] = least_used
            result['context'] = f"Text at {(text_count/total)*100:.0f}% ({text_count}/{total}). Suggesting {least_used}."
            logger.info(result['context'])
            return result

        # RULE 3: Meme check - should appear roughly every 5-7 posts
        meme_count = counts.get('meme', 0)
        if total >= 7 and meme_count == 0:
            result['suggested_type'] = 'meme'
            result['context'] = f"No meme in last {total} posts. Time for one!"
            logger.info(result['context'])
            return result

        # RULE 4: Infographic check - good for educational variety
        infographic_count = counts.get('infographic', 0)
        if total >= 7 and infographic_count == 0:
            result['suggested_type'] = 'infographic'
            result['context'] = f"No infographic in last {total} posts. Suggesting one."
            logger.info(result['context'])
            return result

        # RULE 5: Avoid repeating same type twice in a row
        if len(recent_types) >= 1:
            result['avoid_types'] = [recent_types[0]]

        result['context'] = f"Media distribution OK: {dict(counts)}"
        logger.info(result['context'])
        return result

    def _check_media_budget(self, desired_type: str) -> tuple:
        """
        Checks if we have budget for the desired media type.
        Returns (allowed: bool, fallback_type: str, reason: str)

        Daily limits (to control Vertex AI costs):
        - VIDEO: 1 per day max ($0.50+ each)
        - IMAGE/INFOGRAPHIC/MEME: 5 per day combined ($0.01-0.05 each)
        """
        usage = self._get_daily_media_usage()

        # Video limit: 1 per day
        if desired_type == 'video':
            if usage.get('video', 0) >= 1:
                return (False, 'text', f"Video budget exhausted ({usage['video']}/1 today)")
            return (True, 'video', f"Video OK ({usage['video']}/1 today)")

        # Image types limit: 5 per day combined (increased from 3)
        image_count = usage.get('image', 0) + usage.get('infographic', 0) + usage.get('meme', 0)
        if image_count >= 5:
            return (False, 'text', f"Image budget exhausted ({image_count}/5 today)")

        return (True, desired_type, f"Image budget OK ({image_count}/5 today)")

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

        # Rate limiting: ensure minimum interval between AI calls
        elapsed = time.time() - self._last_ai_call_time
        if elapsed < self._ai_call_min_interval:
            wait_time = self._ai_call_min_interval - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.1f}s before AI call")
            time.sleep(wait_time)
        self._last_ai_call_time = time.time()

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
                        wait_time = (retry + 1) * 5  # Exponential backoff: 5s, 10s
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

    def _ai_select_topic(self, preferred_categories: List[str] = None) -> dict:
        """
        BIG BOSS AI autonomously selects the best topic to comment on.
        Fetches multiple stories and picks the most worthy one.
        """
        # Get multiple stories
        stories = self.news_fetcher.get_multiple_stories(count=5, preferred_categories=preferred_categories)

        if not stories:
            logger.warning("No stories available for AI selection")
            return self._get_trending_story(preferred_categories)

        if len(stories) == 1:
            return stories[0]

        # Build selection prompt
        story_list = "\n".join([
            f"{i+1}. [{s.get('category', 'tech').upper()}] {s['title']}"
            for i, s in enumerate(stories)
        ])

        selection_prompt = f"""Pick the best story for a cynical tech Twitter account.

STORIES:
{story_list}

PICK BASED ON:
- Most interesting to devs/tech people
- Has visual potential (for video/meme)
- Trending or controversial = good
- Avoid boring corporate announcements

Respond with ONLY the number (1-{len(stories)})."""

        try:
            response = self._generate_with_fallback(selection_prompt).strip()
            # Extract number from response
            import re
            match = re.search(r'\d+', response)
            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(stories):
                    selected = stories[idx]
                    logger.info(f"🎖️ BIG BOSS selected: [{selected.get('category', 'tech').upper()}] {selected['title'][:50]}...")
                    return selected

            # Fallback to first story if parsing fails
            logger.warning(f"Could not parse AI selection '{response}', using first story")
            return stories[0]

        except Exception as e:
            logger.warning(f"AI topic selection failed: {e}, falling back to first story")
            return stories[0]

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
        Enforces strict variety and balance to ensure diverse content.

        Rules:
        1. NEVER repeat same category 3+ times in a row
        2. AVOID same category 2 times in a row (strong preference for others)
        3. If any category is >30% of last 10, prefer under-represented ones
        4. Aim for roughly equal distribution across all categories
        """
        all_categories = ['ai', 'crypto', 'finance', 'tech']
        recent_10 = self._get_recent_categories(10)

        if not recent_10:
            return all_categories

        from collections import Counter
        counts = Counter(recent_10)
        total = len(recent_10)

        # RULE 1: HARD BLOCK - Last 2+ posts same category = EXCLUDE that category
        if len(recent_10) >= 2:
            last_category = recent_10[0]
            if recent_10[1] == last_category:
                # Same category twice in a row - MUST switch
                preferred = [cat for cat in all_categories if cat != last_category]
                logger.warning(f"Last 2 posts both '{last_category}'. Forcing switch to: {preferred}")
                # Sort by least used
                preferred.sort(key=lambda cat: counts.get(cat, 0))
                return preferred

        # RULE 2: SOFT AVOID - Last post category gets deprioritized
        last_category = recent_10[0] if recent_10 else None

        # RULE 3: BALANCE CHECK - Stricter threshold (30% instead of 50%)
        if total >= 5:
            for category, count in counts.most_common():
                percentage = (count / total) * 100
                if percentage > 30:
                    # This category is over-represented
                    # Prefer least-used categories, exclude the over-represented one
                    preferred = [cat for cat in all_categories if cat != category]
                    preferred.sort(key=lambda cat: counts.get(cat, 0))
                    logger.info(f"Category '{category}' at {percentage:.0f}% ({count}/{total}). Preferring: {preferred}")
                    return preferred

        # RULE 4: DEFAULT - Prefer least-used categories overall
        # Sort all categories by usage count (ascending)
        preferred = sorted(all_categories, key=lambda cat: counts.get(cat, 0))

        # Move last_category to end if it's first in preferred (soft avoid)
        if last_category and preferred and preferred[0] == last_category:
            preferred.remove(last_category)
            preferred.append(last_category)

        logger.info(f"Category distribution: {dict(counts)}. Preferred order: {preferred}")
        return preferred

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

    def _get_trending_insights(self, category: str = 'ai') -> dict:
        """
        Gets trending insights from influencer analysis.
        Returns style recommendations and trending topics.
        """
        if not self.influencer_analyzer:
            return {'has_data': False}

        try:
            insights = self.influencer_analyzer.get_content_recommendations(category)
            if insights.get('has_data'):
                logger.info(f"Got trending insights for {category}: {insights.get('trending_topics', [])[:5]}")
            return insights
        except Exception as e:
            logger.warning(f"Could not get trending insights: {e}")
            return {'has_data': False}

    def _ai_evaluate_content(self, topic: str, story_context: str, trending_insights: dict) -> dict:
        """
        Uses AI to evaluate if content is worth posting based on trending analysis.
        Returns decision and style recommendations.
        """
        if not self.models:
            return {'should_post': True, 'reason': 'No AI available for evaluation'}

        model = self.models.get(self.model_names[0])
        if not model:
            return {'should_post': True, 'reason': 'No model available'}

        # Build context from trending insights
        trending_context = ""
        if trending_insights.get('has_data'):
            trending_topics = trending_insights.get('trending_topics', [])
            style = trending_insights.get('style_insights', {})
            top_post = trending_insights.get('top_post', {})

            trending_context = f"""
CURRENT TRENDING ANALYSIS:
- Hot topics on Twitter: {', '.join(trending_topics[:10]) if trending_topics else 'N/A'}
- Top performing post style: ~{style.get('avg_length', 150)} chars, {'uses emojis' if style.get('emoji_usage', 0) > 0.3 else 'minimal emojis'}
- Hashtag usage: {'common' if style.get('hashtag_usage', 0) > 0.3 else 'rare'}
- Example high-engagement post: "{top_post.get('text', 'N/A')[:100]}..." ({top_post.get('engagement', 0)} engagement)
"""

        evaluation_prompt = f"""You are a social media strategist. Evaluate if this content is worth posting.

PROPOSED CONTENT:
Topic: {topic}
Context: {story_context[:500]}

{trending_context}

EVALUATE:
1. Is this topic timely and relevant? (trending or newsworthy)
2. Does it align with what's getting engagement on Twitter?
3. Would it interest tech/AI/crypto audience?

Respond in this exact format:
DECISION: POST or SKIP
CONFIDENCE: HIGH/MEDIUM/LOW
REASON: <one line reason>
STYLE_TIP: <one line style recommendation based on trending analysis>
SUGGESTED_HASHTAGS: <2-3 relevant hashtags or "none">
"""

        try:
            response = model.generate_content(evaluation_prompt)
            result_text = response.text.strip()

            # Parse response
            decision = 'POST' in result_text.upper().split('DECISION:')[1].split('\n')[0] if 'DECISION:' in result_text else True
            reason = result_text.split('REASON:')[1].split('\n')[0].strip() if 'REASON:' in result_text else 'AI evaluation'
            style_tip = result_text.split('STYLE_TIP:')[1].split('\n')[0].strip() if 'STYLE_TIP:' in result_text else ''
            hashtags = result_text.split('SUGGESTED_HASHTAGS:')[1].split('\n')[0].strip() if 'SUGGESTED_HASHTAGS:' in result_text else ''

            return {
                'should_post': decision,
                'reason': reason,
                'style_tip': style_tip,
                'hashtags': hashtags if hashtags.lower() != 'none' else '',
                'trending_topics': trending_insights.get('trending_topics', [])
            }
        except Exception as e:
            logger.warning(f"AI evaluation failed: {e}")
            return {'should_post': True, 'reason': f'Evaluation failed: {e}'}

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
        Separated for retry logic. Uses AI to evaluate content based on trending analysis.
        """
        # Get preferred categories based on recent post history (variety + balance)
        preferred_categories = self._get_preferred_categories()
        primary_category = preferred_categories[0] if preferred_categories else 'ai'

        # Fetch trending insights from top influencers
        trending_insights = self._get_trending_insights(primary_category)
        if trending_insights.get('has_data'):
            logger.info(f"Using trending insights: {trending_insights.get('recommendation', 'N/A')}")

        # BIG BOSS AI autonomously selects the topic
        logger.info("🎖️ BIG BOSS selecting intel to comment on...")
        story = self._ai_select_topic(preferred_categories=preferred_categories)
        topic = story['title']
        story_url = story.get('url')  # Real URL or None
        story_context = story.get('context', f"Article: {topic}")  # Rich context from article

        # AI-based content evaluation
        ai_eval = self._ai_evaluate_content(topic, story_context, trending_insights)
        if not ai_eval.get('should_post', True):
            logger.info(f"AI recommends SKIP: {ai_eval.get('reason', 'No reason')}")
            # Try to find better content
            for retry in range(3):
                story = self._get_trending_story(preferred_categories=preferred_categories)
                topic = story['title']
                story_url = story.get('url')
                story_context = story.get('context', f"Article: {topic}")
                ai_eval = self._ai_evaluate_content(topic, story_context, trending_insights)
                if ai_eval.get('should_post', True):
                    logger.info(f"AI approved alternate content: {ai_eval.get('reason', 'N/A')}")
                    break
            else:
                logger.warning("AI rejected all alternatives, proceeding anyway")

        # Store AI recommendations for content generation
        self._current_ai_eval = ai_eval
        logger.info(f"AI style tip: {ai_eval.get('style_tip', 'N/A')}")

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

        # COST-EFFICIENT FORMAT DECISION
        # Check budget FIRST before spending on research (saves API calls)
        if Config.BUDGET_MODE:
            logger.info("BUDGET_MODE enabled, using text-only format")
            post_type = "text"
            research_result = {'format': 'TEXT', 'style_notes': '', 'reasoning': 'Budget mode'}
        else:
            # EARLY BUDGET CHECK - skip expensive research if no budget
            usage = self._get_daily_media_usage()
            image_count = usage.get('image', 0) + usage.get('infographic', 0) + usage.get('meme', 0)

            if image_count >= 5:
                # No media budget - just use text (skip research to save AI calls)
                logger.info(f"💰 Media budget exhausted ({image_count}/5) - using text (saved research API call)")
                post_type = "text"
                research_result = {'format': 'TEXT', 'style_notes': '', 'reasoning': 'Budget exhausted'}
            else:
                # We have budget - do ONE efficient research call
                research_result = {'format': 'TEXT', 'style_notes': '', 'reasoning': 'Default'}

                if self.content_researcher:
                    logger.info("🔬 Researching best content format...")
                    research_result = self.content_researcher.research_topic(
                        topic=topic,
                        context=story_context,
                        category=story.get('category', 'tech')
                    )
                    logger.info(f"Research: {research_result['format']} ({research_result.get('confidence', 'N/A')})")

                    post_type = research_result['format'].lower()
                    confidence = research_result.get('confidence', 'LOW')
                    is_trending = research_result.get('is_trending', False)

                    # For media types, check if worth it (combined into single decision)
                    if post_type != 'text':
                        # HIGH confidence + trending = always use media (no extra AI call)
                        if confidence == 'HIGH' and is_trending:
                            logger.info("✓ HIGH confidence + trending - using media")
                        elif confidence == 'HIGH' or is_trending:
                            # Good enough - use media
                            logger.info(f"✓ {confidence} confidence, trending={is_trending} - using media")
                        else:
                            # LOW/MEDIUM confidence + not trending = save budget, use text
                            logger.info(f"🧠 LOW confidence, not trending - saving budget, using text")
                            post_type = "text"
                else:
                    # No researcher - simple logic (no API call)
                    post_type = "text" if story_url else "infographic"
                    logger.info(f"No researcher - using {'text' if story_url else 'infographic'}")

            logger.info(f"Selected post type: {post_type}")

        strategy = {
            "topic": topic,
            "type": post_type,
            "timestamp": firestore.SERVER_TIMESTAMP
        }

        if post_type == "video":
            # AGENTIC VIDEO: Research-based prompt generation with validation
            logger.info(f"🎬 Creating video for: {topic}")

            style_notes = research_result.get('style_notes', '')
            video_prompt = None

            # Use ContentResearcher to generate and validate video prompt
            if self.content_researcher:
                video_prompt = self.content_researcher.generate_video_prompt(
                    topic=topic,
                    context=story_context,
                    style_notes=style_notes
                )

                if not video_prompt:
                    logger.warning("Could not generate valid video prompt - skipping video")
                    # Return None to skip this post entirely (no fallback to junk)
                    return None

                logger.info(f"Generated video prompt: {video_prompt[:80]}...")
            else:
                # No researcher - generate simple prompt
                video_prompt = f"Futuristic tech visualization about {topic[:50]}, neon lights, data streams, cinematic"

            # Generate caption - dry, cynical style
            caption_prompt = f"""{BIG_BOSS_PERSONA}

Write a SHORT caption for this video.

TOPIC: {topic}

Requirements:
- 60-100 characters
- Dry wit or cynical observation
- NO hashtags, NO emojis
- Creates curiosity

CAPTION:"""

            try:
                caption = self._generate_with_fallback(caption_prompt).strip().strip('"')
                if len(caption) > 150:
                    caption = caption[:147] + "..."
            except Exception as e:
                logger.error(f"Caption generation failed: {e}")
                return None  # Don't post garbage

            # Validate caption makes sense
            if len(caption) < 20 or not caption:
                logger.warning("Caption too short or empty - skipping")
                return None

            # Add URL to caption if available
            if story_url and story_url not in caption:
                if len(caption) + len(story_url) + 4 <= 280:
                    caption = f"{caption}\n\n{story_url}"
                else:
                    max_len = 280 - len(story_url) - 7
                    caption = f"{caption[:max_len]}...\n\n{story_url}"

            strategy["content"] = caption
            strategy["video_prompt"] = video_prompt
            strategy["source_url"] = story_url
            logger.info(f"Video strategy ready: {caption[:50]}...")

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
            # AGENTIC MEME: Research from multiple sources, validate with AI, no fallbacks
            logger.info(f"🔍 Researching memes for: {topic}")
            category = story.get('category', 'tech')

            if not self.meme_fetcher or not self.content_researcher:
                logger.warning("Meme fetcher or researcher not available - skipping")
                return None

            # Research memes from all sources (Reddit, Giphy, Imgflip)
            memes = self.meme_fetcher.research_memes(category, topic)

            if not memes:
                logger.warning("No memes found from any source - skipping post")
                return None  # No fallback - just skip

            # Try to find an approved meme
            approved_meme = None
            meme_local_path = None
            caption = None

            for meme in memes[:3]:  # Check top 3 candidates only (cost efficient - already sorted by score)
                logger.info(f"Evaluating meme: {meme.get('title', '')[:40]}... from {meme.get('source', '')}")

                # AI validates safety and engagement
                validation = self.content_researcher.validate_meme(meme, topic)

                if validation['approved']:
                    logger.info(f"✓ Meme approved: {validation['reason']}")

                    # Try to download
                    meme_local_path = self.meme_fetcher.download_meme(meme['url'])

                    if meme_local_path:
                        approved_meme = meme
                        caption = validation.get('suggested_caption', '')

                        # Generate caption if not provided - dry wit style
                        if not caption:
                            caption_prompt = f"""Write a short, witty caption for this meme.

MEME: {meme.get('title', '')}
TOPIC: {topic}

Requirements: 50-100 chars, dry humor, no hashtags, no emojis.

CAPTION:"""
                            caption = self._generate_with_fallback(caption_prompt).strip().strip('"')

                        break
                    else:
                        logger.warning("Download failed, trying next meme")
                else:
                    logger.debug(f"Meme rejected: {validation['reason']}")

            if not approved_meme:
                logger.warning("No meme passed validation - skipping post entirely")
                return None  # No fallback to junk

            # Add URL to caption
            if story_url and story_url not in caption:
                if len(caption) + len(story_url) + 4 <= 280:
                    caption = f"{caption}\n\n{story_url}"
                else:
                    max_len = 280 - len(story_url) - 7
                    caption = f"{caption[:max_len]}...\n\n{story_url}"

            strategy["content"] = caption
            strategy["meme_local_path"] = meme_local_path
            strategy["meme_source"] = approved_meme.get('permalink', approved_meme.get('source', ''))
            strategy["meme_title"] = approved_meme.get('title', '')
            strategy["source_url"] = story_url
            logger.info(f"Meme strategy ready: {approved_meme.get('title', '')[:40]}...")

        elif post_type == "infographic":
            # AGENTIC INFOGRAPHIC: Validated prompts, no junk generation
            logger.info(f"📊 Creating infographic for: {topic}")

            # Extract key points from article context
            key_points_prompt = f"""Extract 3-5 KEY CONCEPTS from this article for an infographic.

ARTICLE: {story_context[:600]}

Return ONLY a comma-separated list of concepts (2-4 words each):"""

            try:
                key_points_response = self._generate_with_fallback(key_points_prompt)
                key_points = [kp.strip() for kp in key_points_response.split(',')][:5]
                key_points = [kp for kp in key_points if len(kp) > 2]  # Filter empty
                logger.info(f"Key points: {key_points}")
            except Exception as e:
                logger.warning(f"Key points extraction failed: {e}")
                return None  # No fallback

            if len(key_points) < 2:
                logger.warning("Not enough key points extracted - skipping infographic")
                return None

            # Generate and validate infographic prompt
            infographic_visual_prompt = None

            if self.content_researcher:
                infographic_visual_prompt = self.content_researcher.generate_infographic_prompt(
                    topic=topic,
                    context=story_context,
                    key_points=key_points
                )

            if not infographic_visual_prompt:
                # Fallback to simple prompt
                infographic_visual_prompt = f"Clean professional infographic explaining {topic[:50]}. Blue and white color scheme, icons and diagrams, educational visualization, minimalist tech style."

            # Generate caption - informative but casual
            caption_prompt = f"""Write a caption for this infographic.

TOPIC: {topic}
KEY POINTS: {', '.join(key_points)}

Requirements: 80-120 chars, informative but casual, no hashtags, no emojis.

CAPTION:"""

            try:
                caption = self._generate_with_fallback(caption_prompt).strip().strip('"')
                if len(caption) < 20:
                    logger.warning("Caption too short - skipping")
                    return None
            except Exception as e:
                logger.error(f"Caption generation failed: {e}")
                return None

            # Add URL to caption
            if story_url and story_url not in caption:
                if len(caption) + len(story_url) + 4 <= 280:
                    caption = f"{caption}\n\n{story_url}"
                else:
                    max_len = 280 - len(story_url) - 7
                    caption = f"{caption[:max_len]}...\n\n{story_url}"

            strategy["content"] = caption
            strategy["image_prompt"] = infographic_visual_prompt
            strategy["key_points"] = key_points
            strategy["source_url"] = story_url
            logger.info(f"Infographic strategy ready: {infographic_visual_prompt[:50]}...")

        else:
            # Generate text post - dry, cynical style
            logger.info(f"Generating text post for: {topic}")

            if story_url:
                logger.info(f"Using real URL: {story_url}")
                post_prompt = f"""{BIG_BOSS_PERSONA}

Write a tweet about this news.

CONTEXT:
{story_context[:500]}

Topic: "{topic}"
URL: {story_url}

STYLE (pick one):
1. "[Fact]. [Short reaction]."
2. "[What happened]. [Cynical observation]."
3. "[Bold statement]. [Why it matters]."

RULES:
- Under 280 chars total (including URL)
- Include the URL
- Short, punchy, no hype
- NO hashtags, NO emojis

TWEET:
"""
            else:
                logger.info("No URL, generating text-only post")
                post_prompt = f"""{BIG_BOSS_PERSONA}

Write a tweet about: {topic}

STYLE:
- Observation or cynical take
- Short and punchy
- No corporate speak

RULES:
- Under 280 chars
- NO hashtags, NO emojis, NO URLs

TWEET:
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
