"""
Meme Fetcher - Fetches trending memes from free open sources.

Sources:
- Reddit (r/ProgrammerHumor, r/memes, r/cryptocurrency, r/technology)
- Imgflip (popular meme templates)

All free, no API keys required for basic access.
"""

import logging
import requests
import random
import tempfile
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MemeFetcher:
    """Fetches trending memes from free sources for posting."""

    # Reddit subreddits by category
    SUBREDDITS = {
        'ai': ['ProgrammerHumor', 'artificial', 'MachineLearning', 'singularity'],
        'crypto': ['CryptoCurrency', 'Bitcoin', 'ethereum', 'CryptoMemes'],
        'tech': ['ProgrammerHumor', 'technology', 'techhumor', 'softwaregore'],
        'finance': ['wallstreetbets', 'StockMarket', 'investing'],
        'general': ['memes', 'dankmemes', 'me_irl']
    }

    # User agent for Reddit (required)
    HEADERS = {
        'User-Agent': 'PhantomBot/1.0 (Tech News Bot; Contact: bot@example.com)'
    }

    def __init__(self):
        self._cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(minutes=30)

    def fetch_trending_memes(self, category: str = 'tech', limit: int = 10) -> List[Dict]:
        """
        Fetches trending memes from Reddit for a category.
        Returns list of meme dicts with url, title, score, subreddit.
        """
        # Check cache first
        cache_key = f"{category}_{limit}"
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached memes for {category}")
            return self._cache[cache_key]

        subreddits = self.SUBREDDITS.get(category, self.SUBREDDITS['general'])
        all_memes = []

        # Fetch from multiple subreddits
        for subreddit in subreddits[:2]:  # Limit to 2 subreddits to be nice
            try:
                memes = self._fetch_reddit_memes(subreddit, limit=5)
                all_memes.extend(memes)
            except Exception as e:
                logger.warning(f"Failed to fetch from r/{subreddit}: {e}")
                continue

        # Sort by score and filter
        all_memes.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Filter for image posts only
        image_memes = [
            m for m in all_memes
            if self._is_valid_meme_url(m.get('url', ''))
        ]

        result = image_memes[:limit]

        # Cache results
        self._cache[cache_key] = result
        self._cache_time = datetime.now()

        logger.info(f"Fetched {len(result)} memes for {category}")
        return result

    def _fetch_reddit_memes(self, subreddit: str, limit: int = 10) -> List[Dict]:
        """Fetches hot posts from a subreddit."""
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"

        try:
            response = requests.get(url, headers=self.HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()

            memes = []
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})

                # Skip non-image posts, NSFW, stickied
                if post_data.get('over_18') or post_data.get('stickied'):
                    continue

                meme = {
                    'title': post_data.get('title', ''),
                    'url': post_data.get('url', ''),
                    'score': post_data.get('score', 0),
                    'subreddit': subreddit,
                    'permalink': f"https://reddit.com{post_data.get('permalink', '')}",
                    'author': post_data.get('author', ''),
                    'num_comments': post_data.get('num_comments', 0),
                    'created_utc': post_data.get('created_utc', 0)
                }
                memes.append(meme)

            return memes

        except Exception as e:
            logger.error(f"Reddit fetch error for r/{subreddit}: {e}")
            return []

    def _is_valid_meme_url(self, url: str) -> bool:
        """Checks if URL is a valid image/gif for posting."""
        if not url:
            return False

        # Valid image extensions
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']

        # Valid image hosts
        valid_hosts = [
            'i.redd.it',
            'i.imgur.com',
            'imgur.com',
            'preview.redd.it',
            'media.giphy.com',
            'tenor.com'
        ]

        url_lower = url.lower()

        # Check extension
        has_valid_ext = any(url_lower.endswith(ext) for ext in valid_extensions)

        # Check host
        has_valid_host = any(host in url_lower for host in valid_hosts)

        return has_valid_ext or has_valid_host

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is still valid."""
        if cache_key not in self._cache:
            return False
        if self._cache_time is None:
            return False
        return datetime.now() - self._cache_time < self._cache_duration

    def download_meme(self, meme_url: str) -> Optional[str]:
        """
        Downloads meme image to temp file.
        Returns path to downloaded file or None on failure.
        """
        try:
            response = requests.get(meme_url, headers=self.HEADERS, timeout=15)
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

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(response.content)
                file_path = f.name

            file_size = os.path.getsize(file_path)
            logger.info(f"Downloaded meme to {file_path} ({file_size} bytes)")

            # Validate file size (Twitter limit is 5MB for images, 15MB for GIFs)
            max_size = 15 * 1024 * 1024 if ext == '.gif' else 5 * 1024 * 1024
            if file_size > max_size:
                logger.warning(f"Meme too large ({file_size} bytes), skipping")
                os.remove(file_path)
                return None

            return file_path

        except Exception as e:
            logger.error(f"Failed to download meme from {meme_url}: {e}")
            return None

    def get_best_meme_for_topic(self, topic: str, category: str = 'tech') -> Optional[Dict]:
        """
        Gets the best matching meme for a topic.
        Returns meme dict with downloaded file path.
        """
        memes = self.fetch_trending_memes(category, limit=15)

        if not memes:
            logger.warning(f"No memes found for category {category}")
            return None

        # Try to find a meme that matches the topic keywords
        topic_words = set(topic.lower().split())

        # Score memes by relevance
        scored_memes = []
        for meme in memes:
            title_words = set(meme.get('title', '').lower().split())
            overlap = len(topic_words & title_words)
            relevance_score = overlap + (meme.get('score', 0) / 1000)  # Add Reddit score
            scored_memes.append((relevance_score, meme))

        # Sort by relevance
        scored_memes.sort(key=lambda x: x[0], reverse=True)

        # Try to download top memes until one succeeds
        for score, meme in scored_memes[:5]:
            file_path = self.download_meme(meme['url'])
            if file_path:
                meme['local_path'] = file_path
                meme['relevance_score'] = score
                logger.info(f"Selected meme: {meme['title'][:50]}... (score: {score:.1f})")
                return meme

        logger.warning("Could not download any memes")
        return None


class MemeAnalyzer:
    """AI-powered meme analysis for safety and engagement prediction."""

    def __init__(self, generate_func):
        """
        Initialize with a text generation function.
        generate_func should accept a prompt and return text response.
        """
        self.generate = generate_func

    def analyze_meme(self, meme: Dict, topic: str = None) -> Dict:
        """
        Analyzes a meme for safety and engagement potential.
        Returns dict with 'safe', 'engaging', 'reason', 'suggested_caption'.
        """
        title = meme.get('title', '')
        subreddit = meme.get('subreddit', '')
        score = meme.get('score', 0)

        prompt = f"""Analyze this meme for posting on a professional tech Twitter account.

MEME INFO:
- Title: "{title}"
- Source: Reddit r/{subreddit}
- Reddit Score: {score}
- Topic Context: {topic or 'tech news'}

EVALUATE:
1. Is it SAFE for a professional account? (no offensive content, slurs, politics, NSFW)
2. Will it get ENGAGEMENT? (relatable, funny, timely, tech-relevant)
3. Does it fit our BRAND? (tech/AI/crypto focused, witty but professional)

IMPORTANT: Be strict about safety. When in doubt, reject.

Respond in EXACTLY this format:
SAFE: YES or NO
ENGAGING: YES or NO
REASON: <one line explanation>
CAPTION: <suggested tweet caption if safe, or "N/A" if not safe>
"""

        try:
            response = self.generate(prompt)

            # Parse response
            safe = 'SAFE: YES' in response.upper()
            engaging = 'ENGAGING: YES' in response.upper()

            reason = ''
            if 'REASON:' in response:
                reason = response.split('REASON:')[1].split('\n')[0].strip()

            caption = ''
            if 'CAPTION:' in response and safe:
                caption = response.split('CAPTION:')[1].split('\n')[0].strip()
                if caption.upper() == 'N/A':
                    caption = ''

            return {
                'safe': safe,
                'engaging': engaging,
                'reason': reason,
                'suggested_caption': caption,
                'should_post': safe and engaging
            }

        except Exception as e:
            logger.error(f"Meme analysis failed: {e}")
            return {
                'safe': False,
                'engaging': False,
                'reason': f'Analysis failed: {e}',
                'suggested_caption': '',
                'should_post': False
            }


class TrendResearcher:
    """Researches what's trending to inform content strategy."""

    def __init__(self, generate_func, influencer_analyzer=None):
        self.generate = generate_func
        self.influencer = influencer_analyzer

    def get_viral_potential(self, topic: str, context: str = '') -> Dict:
        """
        Analyzes if a topic has viral potential.
        Returns strategy recommendations.
        """
        # Get trending data if available
        trending_context = ""
        if self.influencer:
            try:
                insights = self.influencer.get_content_recommendations('ai')
                if insights.get('has_data'):
                    trending_topics = insights.get('trending_topics', [])
                    trending_context = f"\nCURRENT TRENDING: {', '.join(trending_topics[:10])}"
            except:
                pass

        prompt = f"""Analyze the viral potential of this topic for Twitter.

TOPIC: {topic}
CONTEXT: {context[:500]}{trending_context}

EVALUATE:
1. Is this TRENDING or TIMELY? (happening now, people talking about it)
2. Does it have EMOTIONAL appeal? (surprising, outrageous, inspiring, funny)
3. Would a VIDEO make it MORE viral? (visual story, process, transformation)
4. What STYLE would work best?

VIDEO STYLES that go viral:
- Cyberpunk/futuristic aesthetic
- Anime-inspired (action, dramatic)
- Data visualization / infographic animation
- "Did you know" explainer
- Dramatic reveal / transformation

Respond in this format:
VIRAL_POTENTIAL: HIGH/MEDIUM/LOW
VIDEO_WORTHY: YES or NO
RECOMMENDED_STYLE: <style or "text_only">
HOOK: <one compelling hook line to start the post>
REASON: <why this would/wouldn't go viral>
"""

        try:
            response = self.generate(prompt)

            potential = 'LOW'
            if 'VIRAL_POTENTIAL: HIGH' in response.upper():
                potential = 'HIGH'
            elif 'VIRAL_POTENTIAL: MEDIUM' in response.upper():
                potential = 'MEDIUM'

            video_worthy = 'VIDEO_WORTHY: YES' in response.upper()

            style = 'text_only'
            if 'RECOMMENDED_STYLE:' in response:
                style = response.split('RECOMMENDED_STYLE:')[1].split('\n')[0].strip().lower()

            hook = ''
            if 'HOOK:' in response:
                hook = response.split('HOOK:')[1].split('\n')[0].strip()

            reason = ''
            if 'REASON:' in response:
                reason = response.split('REASON:')[1].split('\n')[0].strip()

            return {
                'viral_potential': potential,
                'video_worthy': video_worthy,
                'recommended_style': style,
                'hook': hook,
                'reason': reason
            }

        except Exception as e:
            logger.error(f"Viral analysis failed: {e}")
            return {
                'viral_potential': 'LOW',
                'video_worthy': False,
                'recommended_style': 'text_only',
                'hook': '',
                'reason': str(e)
            }

    def get_video_prompt_for_style(self, topic: str, style: str, context: str = '') -> str:
        """
        Generates a Veo video prompt based on recommended style.
        """
        style_prompts = {
            'cyberpunk': f"""Cyberpunk futuristic scene: Neon-lit cityscape with holographic displays showing "{topic[:30]}".
Rain-soaked streets, flying vehicles, glowing data streams.
Dark blues and hot pinks, lens flares, cinematic camera movement.
High-tech terminals displaying code and charts. Blade Runner aesthetic.""",

            'anime': f"""Anime-style dramatic sequence about {topic[:30]}.
Dynamic action poses, speed lines, dramatic lighting.
Bold colors, expressive characters, energy effects.
Epic reveal moment with particle effects and dramatic zoom.""",

            'data_viz': f"""Animated data visualization about {topic[:30]}.
3D charts rising from dark surface, glowing data points connecting.
Numbers and statistics floating in space, transforming and updating.
Clean modern aesthetic, smooth camera movements, professional look.""",

            'explainer': f"""Engaging explainer video about {topic[:30]}.
Clean white background with bold graphics appearing.
Step-by-step visual breakdown, icons animating in sequence.
Modern motion graphics, smooth transitions, clear typography.""",

            'transformation': f"""Dramatic before/after transformation showing {topic[:30]}.
Split screen morphing, old vs new comparison.
Satisfying reveal moment, cinematic timing.
Clean professional aesthetic with dramatic lighting."""
        }

        # Default to cyberpunk if style not found (it's eye-catching)
        return style_prompts.get(style, style_prompts['cyberpunk'])
