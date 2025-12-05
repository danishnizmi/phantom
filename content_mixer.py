"""
Global Content Mixer - Combines all content generation capabilities into a unified system.

This module orchestrates:
- News-based posts (Hacker News, RSS feeds)
- YouTube-inspired infographics
- AI-generated videos (Veo 2.0)
- AI-generated images (Imagen 3)
- Meme generation
- Text posts

The mixer decides what type of content to create based on:
- Time of day preferences
- Recent posting history (for variety)
- Available content sources
- Engagement patterns
"""

import logging
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ContentMixer:
    """
    Orchestrates content creation across all available sources and formats.
    Designed to behave like a human content creator mixing different styles.
    """

    def __init__(
        self,
        news_fetcher=None,
        youtube_fetcher=None,
        infographic_generator=None,
        scheduler=None
    ):
        self.news_fetcher = news_fetcher
        self.youtube_fetcher = youtube_fetcher
        self.infographic_generator = infographic_generator
        self.scheduler = scheduler

        # Content type weights (base probabilities)
        # These represent how often each type should appear
        self.content_weights = {
            'news_text': 25,        # Simple news post with link
            'news_image': 20,       # News with AI-generated image
            'news_video': 10,       # News with Veo video (expensive)
            'news_meme': 15,        # Funny/ironic news as meme
            'infographic': 15,      # Educational infographic
            'youtube_explainer': 10, # Based on YouTube trending topics
            'thread': 5,            # Multi-tweet thread
        }

        # Track recent posts to ensure variety
        self.recent_types = []
        self.max_recent = 10

        # Category balance tracking
        self.recent_categories = []

    def _update_history(self, content_type: str, category: str = None):
        """Updates recent posting history for variety enforcement."""
        self.recent_types.append(content_type)
        if len(self.recent_types) > self.max_recent:
            self.recent_types.pop(0)

        if category:
            self.recent_categories.append(category)
            if len(self.recent_categories) > self.max_recent:
                self.recent_categories.pop(0)

    def _get_adjusted_weights(self, preferred_types: List[str] = None) -> Dict[str, float]:
        """
        Adjusts content weights based on:
        - Recent posting history (reduce weight if overused)
        - Time-based preferences from scheduler
        - Available resources
        """
        weights = self.content_weights.copy()

        # 1. Reduce weight for recently used types (variety enforcement)
        for recent_type in self.recent_types[-3:]:  # Last 3 posts
            if recent_type in weights:
                weights[recent_type] = max(5, weights[recent_type] * 0.5)

        # 2. If same type used twice in a row, heavily penalize
        if len(self.recent_types) >= 2 and self.recent_types[-1] == self.recent_types[-2]:
            repeated_type = self.recent_types[-1]
            if repeated_type in weights:
                weights[repeated_type] = 1  # Near-zero chance

        # 3. Apply time-based preferences from scheduler
        if preferred_types:
            for ptype in preferred_types:
                # Boost preferred types
                matching_keys = [k for k in weights.keys() if ptype in k]
                for key in matching_keys:
                    weights[key] = weights[key] * 1.5

        # 4. SMART BUDGET MODE: Reduce expensive media, keep memes and infographics available
        # Video is handled separately in brain.py with daily limits
        from config import Config
        if Config.BUDGET_MODE:
            weights['news_video'] = 0  # Video handled by brain.py, not mixer
            weights['youtube_explainer'] = 0  # Expensive, disable
            # Keep memes and infographics but reduce weight (budget mode allows 2/day total)
            weights['meme'] = max(5, weights.get('meme', 10) * 0.7)
            weights['infographic'] = max(5, weights.get('infographic', 15) * 0.5)
            logger.info("💰 BUDGET_MODE: Reduced media weights, AI decides within limits")

        # 5. Reduce infographic/youtube if fetchers not available
        if not self.youtube_fetcher:
            weights['youtube_explainer'] = 0
            weights['infographic'] = max(5, weights.get('infographic', 15) * 0.7)

        return weights

    def select_content_type(self, preferred_types: List[str] = None) -> str:
        """
        Selects a content type using weighted random selection.
        Ensures variety and respects time preferences.
        """
        weights = self._get_adjusted_weights(preferred_types)

        # Filter out zero-weight options
        valid_options = {k: v for k, v in weights.items() if v > 0}

        if not valid_options:
            logger.warning("No valid content types available, defaulting to news_text")
            return 'news_text'

        # Weighted random selection
        total = sum(valid_options.values())
        r = random.uniform(0, total)

        cumulative = 0
        for content_type, weight in valid_options.items():
            cumulative += weight
            if r <= cumulative:
                logger.info(f"Selected content type: {content_type} (weight: {weight:.1f}/{total:.1f})")
                return content_type

        return 'news_text'  # Fallback

    def get_preferred_category(self) -> List[str]:
        """
        Returns preferred categories based on recent history.
        Ensures category diversity.
        """
        all_categories = ['ai', 'crypto', 'finance', 'tech']

        # Check recent 5 posts for over-representation
        if len(self.recent_categories) >= 5:
            from collections import Counter
            counts = Counter(self.recent_categories[-5:])

            # If any category is >50%, prefer others
            for cat, count in counts.most_common():
                if count >= 3:  # 3+ out of 5 = over-represented
                    preferred = [c for c in all_categories if c != cat]
                    logger.info(f"Category '{cat}' over-represented, preferring: {preferred}")
                    return preferred

        # Default preference order
        return ['ai', 'crypto', 'finance', 'tech']

    def create_content_strategy(self, preferred_types: List[str] = None) -> Dict:
        """
        Creates a complete content strategy based on selected type.
        Returns strategy dict compatible with brain.py format.
        """
        content_type = self.select_content_type(preferred_types)
        preferred_categories = self.get_preferred_category()

        strategy = {
            'mixer_type': content_type,
            'preferred_categories': preferred_categories,
        }

        # Route to appropriate handler
        if content_type == 'infographic':
            strategy.update(self._create_infographic_strategy())
        elif content_type == 'youtube_explainer':
            strategy.update(self._create_youtube_strategy())
        elif content_type.startswith('news_'):
            # Extract sub-type (text, image, video, meme)
            post_format = content_type.replace('news_', '')
            strategy.update({
                'source': 'news',
                'format': post_format,
            })
        elif content_type == 'thread':
            strategy.update({
                'source': 'news',
                'format': 'thread',
            })

        # Update history
        category = strategy.get('category', 'tech')
        self._update_history(content_type, category)

        return strategy

    def _create_infographic_strategy(self) -> Dict:
        """Creates strategy for infographic post."""
        topic = None
        key_points = []
        source_url = None
        context = ""

        # Try YouTube first for educational topics
        if self.youtube_fetcher:
            try:
                video = self.youtube_fetcher.get_infographic_topic()
                if video:
                    topic = video.get('title')
                    key_points = self.youtube_fetcher.extract_key_concepts(video)
                    source_url = video.get('url')
                    context = video.get('description', '')
                    logger.info(f"Infographic topic from YouTube: {topic}")
            except Exception as e:
                logger.warning(f"YouTube fetch failed for infographic: {e}")

        # Fallback to news if no YouTube topic
        if not topic and self.news_fetcher:
            try:
                story = self.news_fetcher.get_trending_story()
                if story:
                    topic = story.get('title')
                    source_url = story.get('url')
                    context = story.get('context', '')
                    logger.info(f"Infographic topic from news: {topic}")
            except Exception as e:
                logger.warning(f"News fetch failed for infographic: {e}")

        return {
            'source': 'youtube' if source_url and 'youtube' in str(source_url) else 'news',
            'format': 'infographic',
            'topic': topic,
            'key_points': key_points,
            'source_url': source_url,
            'context': context,
            'category': self._categorize_topic(topic or ""),
        }

    def _create_youtube_strategy(self) -> Dict:
        """Creates strategy based on YouTube trending topics."""
        if not self.youtube_fetcher:
            # Fallback to regular news
            return {
                'source': 'news',
                'format': 'image',
            }

        try:
            video = self.youtube_fetcher.get_infographic_topic()
            if video:
                return {
                    'source': 'youtube',
                    'format': 'explainer',
                    'topic': video.get('title'),
                    'key_points': self.youtube_fetcher.extract_key_concepts(video),
                    'source_url': video.get('url'),
                    'context': video.get('description', ''),
                    'category': video.get('category', 'tech'),
                    'thumbnail': video.get('thumbnail'),
                }
        except Exception as e:
            logger.warning(f"YouTube strategy creation failed: {e}")

        return {
            'source': 'news',
            'format': 'image',
        }

    def _categorize_topic(self, topic: str) -> str:
        """Categorizes a topic string."""
        topic_lower = topic.lower()

        ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'neural', 'gpt', 'llm']
        crypto_keywords = ['crypto', 'bitcoin', 'ethereum', 'blockchain', 'web3']
        finance_keywords = ['stock', 'market', 'investment', 'trading', 'economy']

        for kw in ai_keywords:
            if kw in topic_lower:
                return 'ai'
        for kw in crypto_keywords:
            if kw in topic_lower:
                return 'crypto'
        for kw in finance_keywords:
            if kw in topic_lower:
                return 'finance'

        return 'tech'


def create_mixed_strategy(
    news_fetcher=None,
    youtube_fetcher=None,
    infographic_generator=None,
    scheduler=None,
    preferred_types: List[str] = None
) -> Dict:
    """
    Convenience function to create a mixed content strategy.
    This is the main entry point for the content mixer.
    """
    mixer = ContentMixer(
        news_fetcher=news_fetcher,
        youtube_fetcher=youtube_fetcher,
        infographic_generator=infographic_generator,
        scheduler=scheduler
    )

    return mixer.create_content_strategy(preferred_types)
