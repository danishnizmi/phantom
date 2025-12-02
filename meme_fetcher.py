"""
Meme Fetcher - Agentic meme discovery from multiple free sources.

Sources (all free, no API keys):
- Reddit (r/ProgrammerHumor, r/memes, etc.)
- Giphy (trending GIFs)
- Imgflip (popular meme templates)

Philosophy: Research first, validate with AI, only post quality content.
If nothing good found, return None - never post junk.
"""

import logging
import requests
import random
import tempfile
import os
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================================
# AI Response Utilities - Robust parsing for AI outputs
# ============================================================================

def parse_ai_field(response: str, field_name: str, default: str = '') -> str:
    """Safely extract a field value from AI response."""
    if not response:
        return default

    patterns = [
        rf'{field_name}:\s*(.+?)(?:\n|$)',
        rf'{field_name}:\s*([^\n]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            value = value.replace('**', '').replace('*', '').strip('"').strip("'")
            if value and value.upper() != 'N/A':
                return value

    return default


def parse_ai_boolean(response: str, field_name: str, default: bool = True) -> bool:
    """Safely extract a boolean field."""
    value = parse_ai_field(response, field_name, '').upper()

    if value in ('YES', 'TRUE', '1', 'Y'):
        return True
    elif value in ('NO', 'FALSE', '0', 'N'):
        return False

    # Check anywhere in response
    pattern = rf'{field_name}[:\s]*(YES|NO)'
    match = re.search(pattern, response.upper())
    if match:
        return match.group(1) == 'YES'

    return default


def clean_ai_prompt(prompt: str, min_length: int = 30) -> Optional[str]:
    """Clean an AI-generated prompt for video/image generation."""
    if not prompt:
        return None

    prefixes = [
        'VIDEO_PROMPT:', 'VIDEO PROMPT:', 'PROMPT:',
        'IMAGE_PROMPT:', 'IMAGE PROMPT:',
        'Here is', "Here's", 'OUTPUT:', 'RESPONSE:',
    ]
    cleaned = prompt.strip()
    for prefix in prefixes:
        if cleaned.upper().startswith(prefix.upper()):
            cleaned = cleaned[len(prefix):].strip()

    cleaned = cleaned.replace('**', '').replace('*', '').replace('`', '')
    cleaned = cleaned.split('\n')[0].strip().strip('"').strip("'")

    if len(cleaned) < min_length:
        return None

    if any(fail in cleaned.upper() for fail in ['CANNOT', 'UNABLE', 'ERROR', 'SORRY']):
        return None

    return cleaned


class MemeSource(ABC):
    """Abstract base class for meme sources."""

    @abstractmethod
    def fetch_memes(self, category: str, limit: int) -> List[Dict]:
        """Fetch memes from this source."""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Return the name of this source."""
        pass


class RedditSource(MemeSource):
    """Fetches memes from Reddit (free, no API key)."""

    SUBREDDITS = {
        'ai': ['ProgrammerHumor', 'artificial', 'MachineLearning', 'singularity', 'ChatGPT'],
        'crypto': ['CryptoCurrency', 'Bitcoin', 'ethereum', 'CryptoMemes', 'SatoshiStreetBets'],
        'tech': ['ProgrammerHumor', 'technology', 'techhumor', 'softwaregore', 'pcmasterrace'],
        'finance': ['wallstreetbets', 'StockMarket', 'mauerstrassenwetten'],
        'general': ['memes', 'dankmemes', 'me_irl', 'AdviceAnimals']
    }

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    def get_source_name(self) -> str:
        return "Reddit"

    def fetch_memes(self, category: str, limit: int = 10) -> List[Dict]:
        subreddits = self.SUBREDDITS.get(category, self.SUBREDDITS['general'])
        all_memes = []

        for subreddit in subreddits[:3]:
            try:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
                response = requests.get(url, headers=self.HEADERS, timeout=10)
                response.raise_for_status()
                data = response.json()

                for post in data.get('data', {}).get('children', []):
                    post_data = post.get('data', {})

                    # Skip NSFW, stickied, non-image
                    if post_data.get('over_18') or post_data.get('stickied'):
                        continue

                    url = post_data.get('url', '')
                    if not self._is_image_url(url):
                        continue

                    meme = {
                        'title': post_data.get('title', ''),
                        'url': url,
                        'score': post_data.get('score', 0),
                        'source': f"Reddit r/{subreddit}",
                        'permalink': f"https://reddit.com{post_data.get('permalink', '')}",
                        'comments': post_data.get('num_comments', 0),
                        'created': post_data.get('created_utc', 0)
                    }
                    all_memes.append(meme)

            except Exception as e:
                logger.debug(f"Reddit r/{subreddit} fetch failed: {e}")
                continue

        return all_memes

    def _is_image_url(self, url: str) -> bool:
        if not url:
            return False
        url_lower = url.lower()
        valid_ext = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        valid_hosts = ['i.redd.it', 'i.imgur.com', 'preview.redd.it']
        return any(url_lower.endswith(ext) for ext in valid_ext) or any(host in url_lower for host in valid_hosts)


class GiphySource(MemeSource):
    """Fetches trending GIFs from Giphy."""

    # Fallback public beta key (rate limited)
    FALLBACK_KEY = "dc6zaTOxFJmzC"
    _api_key = None

    @classmethod
    def _get_api_key(cls) -> str:
        """Get Giphy API key from secrets, fallback to public key."""
        if cls._api_key is None:
            try:
                from config import get_secret
                cls._api_key = get_secret("GIPHY_API_KEY").strip()
                logger.info("Using Giphy API key from secrets")
            except Exception as e:
                logger.debug(f"Giphy secret not found, using public key: {e}")
                cls._api_key = cls.FALLBACK_KEY
        return cls._api_key

    def get_source_name(self) -> str:
        return "Giphy"

    def fetch_memes(self, category: str, limit: int = 10) -> List[Dict]:
        # Map categories to Giphy search terms
        search_terms = {
            'ai': 'artificial intelligence robot',
            'crypto': 'bitcoin cryptocurrency money',
            'tech': 'computer programming code',
            'finance': 'stock market money',
            'general': 'funny reaction'
        }

        query = search_terms.get(category, 'funny tech')
        memes = []
        api_key = self._get_api_key()

        try:
            # Try trending first
            url = f"https://api.giphy.com/v1/gifs/search?api_key={api_key}&q={query}&limit={limit}&rating=g"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            for gif in data.get('data', []):
                images = gif.get('images', {})
                original = images.get('original', {})

                if not original.get('url'):
                    continue

                # Check size (Twitter limit is 15MB for GIFs)
                size = int(original.get('size', 0))
                if size > 10 * 1024 * 1024:  # Skip > 10MB
                    continue

                meme = {
                    'title': gif.get('title', ''),
                    'url': original.get('url'),
                    'score': int(gif.get('trending_datetime', '0') != '0') * 100,  # Trending bonus
                    'source': 'Giphy',
                    'permalink': gif.get('url', ''),
                    'comments': 0,
                    'created': 0
                }
                memes.append(meme)

        except Exception as e:
            logger.debug(f"Giphy fetch failed: {e}")

        return memes


class ImgflipSource(MemeSource):
    """Fetches popular meme templates from Imgflip (free)."""

    def get_source_name(self) -> str:
        return "Imgflip"

    def fetch_memes(self, category: str, limit: int = 10) -> List[Dict]:
        memes = []

        try:
            url = "https://api.imgflip.com/get_memes"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('success'):
                templates = data.get('data', {}).get('memes', [])

                # Filter by category keywords
                category_keywords = {
                    'ai': ['ai', 'robot', 'computer', 'brain', 'think'],
                    'crypto': ['money', 'rich', 'broke', 'trade'],
                    'tech': ['computer', 'code', 'work', 'office', 'think'],
                    'finance': ['money', 'rich', 'broke', 'trade', 'invest'],
                    'general': []  # Accept all
                }

                keywords = category_keywords.get(category, [])

                for template in templates[:50]:  # Check top 50
                    name = template.get('name', '').lower()

                    # If no keywords, accept all. Otherwise filter.
                    if keywords and not any(kw in name for kw in keywords):
                        continue

                    meme = {
                        'title': template.get('name', ''),
                        'url': template.get('url', ''),
                        'score': template.get('box_count', 0) * 10,  # More boxes = more popular
                        'source': 'Imgflip',
                        'permalink': f"https://imgflip.com/meme/{template.get('id', '')}",
                        'comments': 0,
                        'created': 0
                    }
                    memes.append(meme)

                    if len(memes) >= limit:
                        break

        except Exception as e:
            logger.debug(f"Imgflip fetch failed: {e}")

        return memes


class MemeFetcher:
    """
    Aggregates memes from multiple sources.
    Research-first approach - fetches from all sources, ranks by quality.
    """

    def __init__(self):
        self.sources: List[MemeSource] = [
            RedditSource(),
            GiphySource(),
            ImgflipSource()
        ]
        self._cache = {}
        self._cache_time = {}
        self._cache_duration = timedelta(minutes=15)

    def research_memes(self, category: str, topic: str = None) -> List[Dict]:
        """
        Research memes across all sources for a category.
        Returns ranked list of memes from all sources.
        """
        cache_key = f"{category}_{topic or 'general'}"

        # Check cache
        if cache_key in self._cache:
            cache_age = datetime.now() - self._cache_time.get(cache_key, datetime.min)
            if cache_age < self._cache_duration:
                logger.info(f"Using cached memes for {category} ({len(self._cache[cache_key])} memes)")
                return self._cache[cache_key]

        all_memes = []
        sources_tried = []

        for source in self.sources:
            try:
                memes = source.fetch_memes(category, limit=10)
                sources_tried.append(source.get_source_name())
                all_memes.extend(memes)
                logger.info(f"Fetched {len(memes)} memes from {source.get_source_name()}")
            except Exception as e:
                logger.warning(f"Source {source.get_source_name()} failed: {e}")

        if not all_memes:
            logger.warning(f"No memes found from sources: {sources_tried}")
            return []

        # Rank by score
        all_memes.sort(key=lambda x: x.get('score', 0), reverse=True)

        # If topic provided, boost relevance
        if topic:
            topic_words = set(topic.lower().split())
            for meme in all_memes:
                title_words = set(meme.get('title', '').lower().split())
                overlap = len(topic_words & title_words)
                meme['relevance'] = overlap

            # Re-sort with relevance bonus
            all_memes.sort(key=lambda x: x.get('score', 0) + x.get('relevance', 0) * 50, reverse=True)

        # Cache results
        self._cache[cache_key] = all_memes
        self._cache_time[cache_key] = datetime.now()

        logger.info(f"Researched {len(all_memes)} memes from {len(sources_tried)} sources for {category}")
        return all_memes

    def download_meme(self, meme_url: str) -> Optional[str]:
        """Download meme to temp file. Returns path or None."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(meme_url, headers=headers, timeout=15)
            response.raise_for_status()

            # Determine extension
            content_type = response.headers.get('content-type', '')
            if 'gif' in content_type:
                ext = '.gif'
            elif 'png' in content_type:
                ext = '.png'
            elif 'webp' in content_type:
                ext = '.webp'
            else:
                ext = '.jpg'

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(response.content)
                file_path = f.name

            file_size = os.path.getsize(file_path)

            # Twitter limits: 5MB images, 15MB GIFs
            max_size = 15 * 1024 * 1024 if ext == '.gif' else 5 * 1024 * 1024
            if file_size > max_size:
                logger.warning(f"Meme too large ({file_size} bytes), skipping")
                os.remove(file_path)
                return None

            logger.info(f"Downloaded meme: {file_path} ({file_size} bytes)")
            return file_path

        except Exception as e:
            logger.error(f"Download failed for {meme_url}: {e}")
            return None


class ContentResearcher:
    """
    Agentic content researcher - researches trends and decides best content strategy.
    Uses AI to analyze and validate before committing to any content type.
    """

    def __init__(self, generate_func, influencer_analyzer=None):
        """
        Args:
            generate_func: Function to generate text (AI model call)
            influencer_analyzer: Optional InfluencerAnalyzer for Twitter trends
        """
        self.generate = generate_func
        self.influencer = influencer_analyzer
        self._research_cache = {}
        self._cache_time = None

    def research_topic(self, topic: str, context: str, category: str) -> Dict:
        """
        AI researches topic and decides best content format dynamically.
        Returns dict with format, style, and reasoning.
        """
        # Get trending context if available
        trending_context = self._get_trending_context(category)

        prompt = f"""You're an AI running a tech Twitter account. Decide the best content format for this topic.

TOPIC: {topic}
CATEGORY: {category}
CONTEXT: {context[:500]}
{f'CURRENT TRENDS: {trending_context}' if trending_context else ''}

AVAILABLE FORMATS:
- VIDEO: AI-generated visuals (best for high-impact, visual topics)
- MEME: Fetched from internet (best for irony, relatable content)
- INFOGRAPHIC: Educational visuals (best for explanations, data)
- TEXT: Link + caption (best for quick news, simple updates)

THINK ABOUT:
1. Does this topic have strong VISUAL potential? → VIDEO
2. Is there irony, drama, or relatable frustration? → MEME
3. Are there concepts to explain or compare? → INFOGRAPHIC
4. Is it just news that needs fast sharing? → TEXT

Make your own judgment. Don't default to TEXT - media gets more engagement.

Respond:
RECOMMENDED_FORMAT: <VIDEO|MEME|INFOGRAPHIC|TEXT>
CONFIDENCE: <HIGH|MEDIUM|LOW>
REASONING: <one line - why this format>
STYLE_NOTES: <if VIDEO/MEME: style direction. If TEXT: N/A>
IS_TRENDING: <YES|NO>
"""

        try:
            response = self.generate(prompt)

            # Use robust parsing utilities
            fmt = parse_ai_field(response, 'RECOMMENDED_FORMAT', 'TEXT').upper()
            if fmt not in ['TEXT', 'MEME', 'INFOGRAPHIC', 'VIDEO']:
                fmt = 'TEXT'

            conf = parse_ai_field(response, 'CONFIDENCE', 'LOW').upper()
            if conf not in ['HIGH', 'MEDIUM', 'LOW']:
                conf = 'LOW'

            result = {
                'format': fmt,
                'confidence': conf,
                'reasoning': parse_ai_field(response, 'REASONING', ''),
                'style_notes': parse_ai_field(response, 'STYLE_NOTES', ''),
                'is_trending': parse_ai_boolean(response, 'IS_TRENDING', False)
            }

            logger.info(f"Research result: {result['format']} ({result['confidence']}) - {result['reasoning'][:50] if result['reasoning'] else 'N/A'}")
            return result

        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {'format': 'TEXT', 'confidence': 'LOW', 'reasoning': str(e), 'style_notes': '', 'is_trending': False}

    def _get_trending_context(self, category: str) -> str:
        """Get trending topics from influencer analyzer if available."""
        if not self.influencer:
            return ""

        try:
            insights = self.influencer.get_content_recommendations(category)
            if insights.get('has_data'):
                topics = insights.get('trending_topics', [])[:5]
                return ', '.join(topics)
        except Exception as e:
            logger.debug(f"Could not get trending context: {e}")

        return ""

    def validate_meme(self, meme: Dict, topic: str) -> Dict:
        """
        AI validates if a meme is safe and engaging for posting.
        Returns dict with 'approved', 'reason', 'suggested_caption' (Big Boss style).
        """
        title = meme.get('title', '')
        source = meme.get('source', '')
        score = meme.get('score', 0)

        prompt = f"""Evaluate this meme for a cynical tech Twitter account.

MEME:
- Title: "{title}"
- Source: {source}
- Score: {score}
- Topic: {topic}

CHECK:
1. SAFE? (No politics, NSFW, slurs, controversial)
2. RELEVANT? (Tech/AI/crypto/finance audience)
3. FUNNY? (Actually good, not cringe)

Be strict. Skip anything questionable.

If APPROVED, write a SHORT caption (50-100 chars):
- Dry wit, cynical observation
- No emojis, no hashtags
- Sound like a tired dev, not a marketer

Respond:
APPROVED: YES or NO
REASON: <why>
SUGGESTED_CAPTION: <caption or N/A>
"""

        try:
            response = self.generate(prompt)

            # Use robust parsing utilities
            approved = parse_ai_boolean(response, 'APPROVED', default=False)
            reason = parse_ai_field(response, 'REASON', 'No reason provided')
            caption = parse_ai_field(response, 'SUGGESTED_CAPTION', '') if approved else ''

            return {
                'approved': approved,
                'reason': reason,
                'suggested_caption': caption
            }

        except Exception as e:
            logger.error(f"Meme validation failed: {e}")
            return {'approved': False, 'reason': str(e), 'suggested_caption': ''}

    def generate_video_prompt(self, topic: str, context: str, style_notes: str) -> Optional[str]:
        """
        Generate a CINEMATIC video prompt like a film director.
        Focus on visual storytelling, camera work, lighting, and mood - NOT literal news visualization.

        Returns artistically crafted prompt optimized for Veo 3.
        """
        # Extract emotional/thematic essence, not literal keywords
        topic_lower = topic.lower()

        # Determine MOOD and VISUAL METAPHOR (not literal representation)
        if any(kw in topic_lower for kw in ['surge', 'rally', 'soar', 'boom', 'record']):
            mood = "triumphant, ascending, powerful"
            metaphor = "rising, breaking through, reaching new heights"
        elif any(kw in topic_lower for kw in ['crash', 'fall', 'plunge', 'fear', 'crisis']):
            mood = "tense, dramatic, uncertain"
            metaphor = "falling, shattering, fragile balance"
        elif any(kw in topic_lower for kw in ['launch', 'release', 'unveil', 'announce', 'new']):
            mood = "anticipation, reveal, dawn of something new"
            metaphor = "emergence, birth, first light"
        elif any(kw in topic_lower for kw in ['battle', 'compete', 'challenge', 'versus', 'fight']):
            mood = "confrontation, tension, high stakes"
            metaphor = "clash, standoff, opposing forces"
        elif any(kw in topic_lower for kw in ['future', 'ai', 'robot', 'autonomous']):
            mood = "wonder, possibility, technological sublime"
            metaphor = "awakening, consciousness, infinite potential"
        else:
            mood = "intrigue, discovery, transformation"
            metaphor = "journey, change, evolution"

        prompt = f"""You are a visionary AI FILM DIRECTOR. Create a stunning, artistic VIDEO PROMPT for Veo 3.

INSPIRATION: {topic}
MOOD TO EVOKE: {mood}
VISUAL METAPHOR: {metaphor}

CRITICAL RULES - Think like a FILMMAKER, not a news illustrator:
1. NEVER literally show logos, text, or news graphics
2. NEVER describe "a person reading news" or "stock charts"
3. CREATE abstract, artistic, CINEMATIC visuals that EVOKE the feeling
4. Use FILMMAKING LANGUAGE: camera movements, lighting, composition

REQUIRED ELEMENTS IN YOUR PROMPT:
- CAMERA: Specific movement (slow dolly, crane up, tracking shot, push-in, orbiting, handheld)
- LIGHTING: Dramatic lighting (volumetric rays, rim light, silhouette, golden hour, neon glow, chiaroscuro)
- SUBJECT: Abstract or metaphorical visual (NOT literal news content)
- ATMOSPHERE: Mood elements (particles, mist, rain, light rays, reflections)
- STYLE: Cinematic quality terms (anamorphic, shallow depth of field, 4K, filmic grain)

GREAT EXAMPLES (study these):
- "Slow tracking shot through an abandoned server room, dust particles floating in volumetric light beams, cables hanging like vines, a single monitor flickers to life in the distance, anamorphic lens flare, cyberpunk atmosphere"
- "Crane shot rising above an infinite mirror maze, reflections fragmenting into thousands of copies, golden hour light streaming through, dreamlike and surreal, shallow depth of field"
- "Extreme close-up of a water droplet falling in slow motion, inside it we see a miniature city skyline reflected, the drop shatters on impact revealing a burst of light, macro photography style"
- "A lone figure stands at the edge of a vast digital ocean, waves made of glowing data particles, camera slowly orbits as the figure reaches toward the horizon, silhouette against bioluminescent blue"

NOW CREATE YOUR CINEMATIC PROMPT (200-300 chars, pure visual poetry, no labels):"""

        try:
            response = self.generate(prompt)

            # Use robust cleaning utility
            video_prompt = clean_ai_prompt(response, min_length=50)

            # Validate it sounds cinematic, not literal
            if video_prompt:
                literal_fails = ['logo', 'headline', 'news', 'article', 'stock chart', 'graph showing', 'text reading']
                if any(fail in video_prompt.lower() for fail in literal_fails):
                    logger.warning("Prompt too literal, regenerating with artistic focus")
                    video_prompt = None

            # If cleaning failed or too literal, use artistic fallback
            if not video_prompt:
                logger.warning(f"Using artistic fallback prompt")
                fallbacks = [
                    "Slow push-in through layers of translucent geometric shapes, each layer glowing with different colors, particles floating, volumetric light rays, ethereal atmosphere, shallow depth of field",
                    "Drone shot descending through clouds into a vast crystalline landscape, light refracting into rainbows, mist rolling across mirror-like surfaces, sunrise colors, cinematic and dreamlike",
                    "Tracking shot following a single glowing orb traveling through an infinite dark space, leaving trails of light, other orbs awakening as it passes, cosmic scale, anamorphic lens",
                    "Close-up of liquid metal morphing and flowing, reflecting a futuristic cityscape, camera slowly pulls back to reveal impossible architecture, chrome and neon, blade runner aesthetic",
                ]
                video_prompt = random.choice(fallbacks)

            logger.info(f"Generated video prompt: {video_prompt[:80]}...")
            return video_prompt

        except Exception as e:
            logger.error(f"Video prompt generation failed: {e}")
            # Return artistic default
            return "Slow dolly through an abstract digital landscape, geometric shapes floating in volumetric light, particles drifting, camera reveals infinite depth, cinematic and ethereal"

    def generate_infographic_prompt(self, topic: str, context: str, key_points: List[str]) -> Optional[str]:
        """
        Generate an infographic prompt with built-in self-validation (single AI call).
        Returns validated prompt or None.
        """
        points_text = '\n'.join(f"- {p}" for p in key_points[:5])

        # Combined generation + validation in ONE call to save API costs
        prompt = f"""Create an INFOGRAPHIC image prompt for this topic. Self-validate before responding.

TOPIC: {topic}
KEY POINTS:
{points_text}

INFOGRAPHIC STYLES:
- Flowchart: Process steps connected by arrows
- Comparison: Side-by-side boxes showing differences
- Timeline: Chronological progression
- Stats: Big numbers with icons
- Diagram: System architecture or concept map

REQUIREMENTS (self-validate):
1. SPECIFIC enough for image generation
2. CLEAN, PROFESSIONAL, EDUCATIONAL style
3. RELEVANT to topic and key points
4. 80-150 characters

Only respond with the final prompt. If you can't create a good one, respond "CANNOT_GENERATE".

INFOGRAPHIC_PROMPT:"""

        try:
            response = self.generate(prompt)

            # Use robust cleaning utility
            infographic_prompt = clean_ai_prompt(response, min_length=30)

            if not infographic_prompt:
                logger.warning("AI could not generate valid infographic prompt")
                # Return a sensible fallback based on topic
                infographic_prompt = f"Professional tech infographic about {topic[:40]}, clean design, educational diagram, modern style"

            return infographic_prompt

        except Exception as e:
            logger.error(f"Infographic prompt generation failed: {e}")
            return None
