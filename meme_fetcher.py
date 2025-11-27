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
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


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

            result = {
                'format': 'TEXT',
                'confidence': 'LOW',
                'reasoning': '',
                'style_notes': '',
                'is_trending': False
            }

            if 'RECOMMENDED_FORMAT:' in response:
                fmt = response.split('RECOMMENDED_FORMAT:')[1].split('\n')[0].strip().upper()
                if fmt in ['TEXT', 'MEME', 'INFOGRAPHIC', 'VIDEO']:
                    result['format'] = fmt

            if 'CONFIDENCE:' in response:
                conf = response.split('CONFIDENCE:')[1].split('\n')[0].strip().upper()
                if conf in ['HIGH', 'MEDIUM', 'LOW']:
                    result['confidence'] = conf

            if 'REASONING:' in response:
                result['reasoning'] = response.split('REASONING:')[1].split('\n')[0].strip()

            if 'STYLE_NOTES:' in response:
                result['style_notes'] = response.split('STYLE_NOTES:')[1].split('\n')[0].strip()

            if 'IS_TRENDING:' in response:
                result['is_trending'] = 'YES' in response.split('IS_TRENDING:')[1].split('\n')[0].upper()

            logger.info(f"Research result: {result['format']} ({result['confidence']}) - {result['reasoning'][:50]}")
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

            approved = 'APPROVED: YES' in response.upper()
            reason = ''
            caption = ''

            if 'REASON:' in response:
                reason = response.split('REASON:')[1].split('\n')[0].strip()

            if approved and 'SUGGESTED_CAPTION:' in response:
                caption = response.split('SUGGESTED_CAPTION:')[1].split('\n')[0].strip()
                if caption.upper() == 'N/A':
                    caption = ''

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
        Generate a creative video prompt. AI decides the best visual style dynamically.
        Returns validated prompt or None if can't create good one.
        """
        # Extract key concepts from topic for better prompts
        topic_lower = topic.lower()

        # Determine visual theme based on topic keywords
        if any(kw in topic_lower for kw in ['bitcoin', 'crypto', 'blockchain', 'token', 'defi']):
            theme_hint = "cryptocurrency, digital gold, blockchain networks, neon green data streams"
        elif any(kw in topic_lower for kw in ['ai', 'artificial', 'gemini', 'gpt', 'model', 'neural']):
            theme_hint = "artificial intelligence, neural networks, glowing circuits, futuristic technology"
        elif any(kw in topic_lower for kw in ['stock', 'market', 'invest', 'fund', 'billion']):
            theme_hint = "financial markets, stock tickers, trading floors, money flow visualization"
        elif any(kw in topic_lower for kw in ['apple', 'google', 'meta', 'microsoft', 'amazon']):
            theme_hint = "tech giant headquarters, sleek product design, corporate innovation"
        else:
            theme_hint = "technology, innovation, digital transformation, futuristic visualization"

        prompt = f"""Create a VIDEO PROMPT for an AI video generator (like Veo). Output ONLY the prompt text, nothing else.

TOPIC: {topic}
THEME: {theme_hint}
{f'STYLE DIRECTION: {style_notes}' if style_notes else ''}

Write a cinematic video description (100-150 chars) that includes:
- A specific visual scene (not abstract concepts)
- Lighting and color palette
- Camera movement or visual flow
- Modern, striking aesthetic

EXAMPLE OUTPUTS:
- "Glowing blockchain network expanding across dark space, neon green data packets flowing between nodes, cinematic zoom out"
- "Futuristic AI chip pulsing with blue light, neural pathways branching outward, dramatic lens flare, 4K quality"
- "Stock market holographic display, green numbers rising, trader silhouette watching, cyberpunk city background"

YOUR VIDEO PROMPT (just the description, no labels):"""

        try:
            response = self.generate(prompt)

            # Clean up response - remove any labels, quotes, markdown
            video_prompt = response.strip()

            # Remove common prefixes the AI might add
            remove_prefixes = ['VIDEO_PROMPT:', 'VIDEO PROMPT:', 'PROMPT:', 'Here is', 'Here\'s', '**', '*']
            for prefix in remove_prefixes:
                if video_prompt.upper().startswith(prefix.upper()):
                    video_prompt = video_prompt[len(prefix):].strip()

            # Remove markdown formatting
            video_prompt = video_prompt.replace('**', '').replace('*', '').strip()

            # Get first line only, remove quotes
            video_prompt = video_prompt.split('\n')[0].strip().strip('"').strip("'")

            # Final validation
            if len(video_prompt) < 30:
                logger.warning(f"Video prompt too short ({len(video_prompt)} chars): {video_prompt}")
                # Generate a fallback based on theme
                video_prompt = f"Cinematic visualization of {theme_hint}, dramatic lighting, futuristic aesthetic, 4K quality"
                logger.info(f"Using fallback video prompt: {video_prompt[:50]}...")

            if 'CANNOT' in video_prompt.upper() or len(video_prompt) < 30:
                logger.warning("AI could not generate valid video prompt, using default")
                video_prompt = f"Cinematic visualization of {theme_hint}, neon lights, data streams, dramatic camera movement"

            logger.info(f"Generated video prompt: {video_prompt[:80]}...")
            return video_prompt

        except Exception as e:
            logger.error(f"Video prompt generation failed: {e}")
            # Return a reasonable default instead of None
            return f"Futuristic tech visualization, {theme_hint}, cinematic lighting, dramatic atmosphere"

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

            if 'CANNOT_GENERATE' in response.upper():
                logger.warning("AI could not generate valid infographic prompt")
                return None

            if 'INFOGRAPHIC_PROMPT:' in response:
                infographic_prompt = response.split('INFOGRAPHIC_PROMPT:')[1].strip()
            else:
                infographic_prompt = response.strip()

            infographic_prompt = infographic_prompt.split('\n')[0].strip().strip('"')

            if len(infographic_prompt) < 30:
                logger.warning(f"Infographic prompt too short")
                return None

            return infographic_prompt

        except Exception as e:
            logger.error(f"Infographic prompt generation failed: {e}")
            return None
