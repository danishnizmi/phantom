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
    """Fetches trending GIFs from Giphy (free tier)."""

    # Giphy public beta key (rate limited but free)
    PUBLIC_KEY = "dc6zaTOxFJmzC"

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

        try:
            # Try trending first
            url = f"https://api.giphy.com/v1/gifs/search?api_key={self.PUBLIC_KEY}&q={query}&limit={limit}&rating=g"
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
        Research a topic and return content recommendations.
        Returns dict with format, style, and reasoning.
        """
        # Get trending context if available
        trending_context = self._get_trending_context(category)

        prompt = f"""You are a content strategist for a tech Twitter account. Analyze this topic and recommend the BEST content format.

TOPIC: {topic}
CATEGORY: {category}
ARTICLE CONTEXT: {context[:800]}
{f'CURRENTLY TRENDING: {trending_context}' if trending_context else ''}

AVAILABLE FORMATS:
1. TEXT - Simple tweet with article link (Twitter shows preview card)
2. MEME - Funny/ironic image from internet + witty caption
3. INFOGRAPHIC - Educational visual explaining concepts
4. VIDEO - Animated visual (expensive, use sparingly for HIGH impact topics)

DECISION CRITERIA:
- TEXT: Default for news with good link preview. Low effort, still effective.
- MEME: Story is ironic, absurd, or relatable frustration. Community will appreciate humor.
- INFOGRAPHIC: Educational content, comparisons, stats, "how it works" topics.
- VIDEO: Only for MAJOR announcements or highly visual processes. Very expensive.

Analyze and respond in this EXACT format:
RECOMMENDED_FORMAT: <TEXT|MEME|INFOGRAPHIC|VIDEO>
CONFIDENCE: <HIGH|MEDIUM|LOW>
REASONING: <one line explaining why>
STYLE_NOTES: <specific style guidance if MEME/INFOGRAPHIC/VIDEO, or "N/A" for TEXT>
IS_TRENDING: <YES|NO - is this topic currently hot?>
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
        Returns dict with 'approved', 'reason', 'suggested_caption'.
        """
        title = meme.get('title', '')
        source = meme.get('source', '')
        score = meme.get('score', 0)

        prompt = f"""Evaluate this meme for a professional tech Twitter account.

MEME:
- Title: "{title}"
- Source: {source}
- Popularity Score: {score}
- Topic Context: {topic}

EVALUATE:
1. SAFE? (No offensive content, politics, NSFW, slurs, controversial takes)
2. RELEVANT? (Fits tech/AI/crypto/finance audience)
3. ENGAGING? (Actually funny/relatable, not cringe)

Be STRICT. When in doubt, reject. We'd rather post nothing than something bad.

Respond EXACTLY:
APPROVED: YES or NO
REASON: <one line>
SUGGESTED_CAPTION: <witty caption if approved, "N/A" if not>
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

    def validate_prompt(self, prompt_type: str, prompt_text: str, topic: str) -> Dict:
        """
        AI validates if a generation prompt makes sense before using it.
        Prevents wasting API calls on bad prompts.
        """
        validation_prompt = f"""Evaluate this {prompt_type} generation prompt.

PROMPT TO EVALUATE:
"{prompt_text}"

TOPIC CONTEXT: {topic}

CHECK:
1. Is it SPECIFIC enough? (Not vague like "cool tech stuff")
2. Is it ACHIEVABLE? (AI can actually generate this)
3. Is it RELEVANT? (Matches the topic)
4. Is it PROFESSIONAL? (Appropriate for business account)

Respond EXACTLY:
VALID: YES or NO
ISSUES: <list any problems, or "None">
IMPROVED_PROMPT: <better version if needed, or "N/A" if already good>
"""

        try:
            response = self.generate(validation_prompt)

            valid = 'VALID: YES' in response.upper()
            issues = ''
            improved = ''

            if 'ISSUES:' in response:
                issues = response.split('ISSUES:')[1].split('\n')[0].strip()

            if 'IMPROVED_PROMPT:' in response:
                improved = response.split('IMPROVED_PROMPT:')[1].split('\n')[0].strip()
                if improved.upper() == 'N/A':
                    improved = prompt_text

            return {
                'valid': valid,
                'issues': issues,
                'improved_prompt': improved if improved else prompt_text
            }

        except Exception as e:
            logger.error(f"Prompt validation failed: {e}")
            return {'valid': True, 'issues': '', 'improved_prompt': prompt_text}

    def generate_video_prompt(self, topic: str, context: str, style_notes: str) -> Optional[str]:
        """
        Generate and validate a video prompt.
        Returns validated prompt or None if can't create good one.
        """
        prompt = f"""Create a VIDEO generation prompt for this topic.

TOPIC: {topic}
CONTEXT: {context[:500]}
STYLE GUIDANCE: {style_notes}

VIDEO STYLES THAT WORK:
- Cyberpunk: Neon cities, holograms, rain, pink/blue lights
- Data visualization: 3D charts, floating numbers, clean aesthetic
- Tech montage: Code flowing, circuits lighting up, futuristic UI

Create a SPECIFIC, VISUAL prompt (100-200 chars) that Veo can actually generate.

VIDEO_PROMPT:"""

        try:
            response = self.generate(prompt)

            if 'VIDEO_PROMPT:' in response:
                video_prompt = response.split('VIDEO_PROMPT:')[1].strip()
            else:
                video_prompt = response.strip()

            # Clean up
            video_prompt = video_prompt.split('\n')[0].strip().strip('"')

            if len(video_prompt) < 30:
                logger.warning(f"Video prompt too short: {video_prompt}")
                return None

            # Validate the prompt
            validation = self.validate_prompt('video', video_prompt, topic)
            if not validation['valid']:
                logger.warning(f"Video prompt invalid: {validation['issues']}")
                video_prompt = validation['improved_prompt']

            return video_prompt

        except Exception as e:
            logger.error(f"Video prompt generation failed: {e}")
            return None

    def generate_infographic_prompt(self, topic: str, context: str, key_points: List[str]) -> Optional[str]:
        """
        Generate and validate an infographic prompt.
        Returns validated prompt or None.
        """
        points_text = '\n'.join(f"- {p}" for p in key_points[:5])

        prompt = f"""Create an INFOGRAPHIC image prompt for this topic.

TOPIC: {topic}
KEY POINTS:
{points_text}

INFOGRAPHIC STYLES:
- Flowchart: Process steps connected by arrows
- Comparison: Side-by-side boxes showing differences
- Timeline: Chronological progression
- Stats: Big numbers with icons
- Diagram: System architecture or concept map

Create a SPECIFIC prompt (80-150 chars) for Imagen to generate.
Focus on CLEAN, PROFESSIONAL, EDUCATIONAL visual.

INFOGRAPHIC_PROMPT:"""

        try:
            response = self.generate(prompt)

            if 'INFOGRAPHIC_PROMPT:' in response:
                infographic_prompt = response.split('INFOGRAPHIC_PROMPT:')[1].strip()
            else:
                infographic_prompt = response.strip()

            infographic_prompt = infographic_prompt.split('\n')[0].strip().strip('"')

            if len(infographic_prompt) < 30:
                logger.warning(f"Infographic prompt too short")
                return None

            # Validate
            validation = self.validate_prompt('infographic', infographic_prompt, topic)
            if not validation['valid']:
                infographic_prompt = validation['improved_prompt']

            return infographic_prompt

        except Exception as e:
            logger.error(f"Infographic prompt generation failed: {e}")
            return None
