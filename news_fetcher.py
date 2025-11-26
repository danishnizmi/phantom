import logging
import requests
import os
from typing import Dict, List, Optional
import random
import feedparser

logger = logging.getLogger(__name__)

# Optional: newspaper3k for better article extraction
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    logger.debug("newspaper3k not available, using basic validation")


class NewsFetcher:
    """Fetches real tech news from various sources including AI, crypto, and finance."""

    @staticmethod
    def _build_spam_patterns() -> List[str]:
        """
        Dynamically generates spam patterns instead of hardcoding.
        Includes fake AI model names, marketing spam, etc.
        """
        patterns = []

        # 1. Generate fake AI model names dynamically
        # Current real versions: GPT-4, Claude 3, Gemini 2
        # Future versions don't exist yet
        ai_brands = ['gpt', 'claude', 'gemini', 'llama', 'mistral']
        for brand in ai_brands:
            # Versions that don't exist (future versions)
            for version in range(4, 10):  # 4-9 are future/fake
                if brand == 'gpt' and version <= 5:
                    patterns.append(f'{brand}-{version}')  # gpt-5+ is fake
                elif brand == 'claude' and version >= 4:
                    patterns.append(f'{brand} {version}')  # claude 4+ is fake
                elif brand == 'gemini' and version >= 3:
                    patterns.append(f'{brand} {version}')  # gemini 3+ is fake

        # 2. Known fake product patterns
        patterns.extend([
            'nano banana', 'banana pro',  # Known spam products
        ])

        # 3. Marketing spam phrases
        marketing_spam = [
            'free trial', 'limited time offer', 'act now',
            'click here', 'buy now', '100% free', 'make money',
            'exclusive deal', 'hurry up', 'don\'t miss',
        ]
        patterns.extend(marketing_spam)

        # 4. Load additional patterns from environment (comma-separated)
        extra_patterns = os.getenv('SPAM_PATTERNS', '')
        if extra_patterns:
            patterns.extend([p.strip().lower() for p in extra_patterns.split(',') if p.strip()])

        return patterns

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        # Build spam patterns dynamically
        self.spam_patterns = self._build_spam_patterns()
        logger.info(f"Loaded {len(self.spam_patterns)} spam patterns (dynamic)")

        # RSS feeds for different categories
        self.feeds = {
            'ai': [
                'https://blog.google/technology/ai/rss/',
                'https://openai.com/blog/rss.xml',
                'https://www.anthropic.com/news/rss.xml',
            ],
            'crypto': [
                'https://cointelegraph.com/rss',
                'https://www.coindesk.com/arc/outboundfeeds/rss/',
            ],
            'finance': [
                'https://finance.yahoo.com/news/rssindex',
                'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            ]
        }

    def _has_spam_pattern(self, title: str) -> bool:
        """Check if title contains dynamically generated spam patterns."""
        if not title:
            return True  # No title = suspicious

        title_lower = title.lower()
        for pattern in self.spam_patterns:
            if pattern in title_lower:
                logger.debug(f"Spam pattern '{pattern}' found in: {title[:50]}...")
                return True
        return False

    def _validate_with_newspaper(self, url: str) -> bool:
        """
        Use newspaper3k to validate article (if available).
        Returns True if article seems legitimate.
        """
        if not NEWSPAPER_AVAILABLE:
            return True  # Skip validation if library not available

        try:
            article = Article(url)
            article.download()
            article.parse()

            # Basic validation: must have title and some text
            if not article.title or len(article.title) < 10:
                return False
            if not article.text or len(article.text) < 100:
                return False

            return True
        except Exception as e:
            logger.debug(f"newspaper3k validation failed for {url}: {e}")
            return True  # Don't block on validation errors

    def _is_valid_story(self, title: str, url: str) -> bool:
        """
        Lightweight validation - checks if story is legitimate.
        Uses title pattern matching (not strict domain filtering).
        """
        # Must have a URL
        if not url:
            return False

        # Title sanity checks
        if not title or len(title) < 10:
            return False
        if len(title) > 300:  # Suspiciously long
            return False

        # Check for spam patterns in title
        if self._has_spam_pattern(title):
            return False

        return True

    def fetch_hacker_news_top_stories(self, limit: int = 30) -> List[Dict]:
        """
        Fetches top stories from Hacker News API.
        Returns list of {title, url, score, id} dicts.
        """
        try:
            # Get top story IDs
            response = self.session.get(
                'https://hacker-news.firebaseio.com/v0/topstories.json',
                timeout=10
            )
            response.raise_for_status()
            story_ids = response.json()[:limit]

            stories = []
            for story_id in story_ids[:15]:  # Get first 15 for speed
                try:
                    story_response = self.session.get(
                        f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json',
                        timeout=5
                    )
                    story_data = story_response.json()

                    # Only include stories with URLs (not Ask HN, etc.)
                    if story_data and story_data.get('url'):
                        stories.append({
                            'title': story_data.get('title', ''),
                            'url': story_data['url'],
                            'score': story_data.get('score', 0),
                            'id': story_id,
                            'source': 'HackerNews'
                        })
                except Exception as e:
                    logger.warning(f"Failed to fetch HN story {story_id}: {e}")
                    continue

            logger.info(f"✓ Fetched {len(stories)} stories from Hacker News")
            return stories

        except Exception as e:
            logger.error(f"Failed to fetch Hacker News: {e}")
            return []

    def fetch_rss_feed(self, feed_url: str, category: str, limit: int = 10) -> List[Dict]:
        """
        Fetches stories from an RSS feed.
        Returns list of {title, url, source, category} dicts.
        """
        stories = []
        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:limit]:
                if hasattr(entry, 'link') and hasattr(entry, 'title'):
                    # Validate story before adding
                    if self._is_valid_story(entry.title, entry.link):
                        stories.append({
                            'title': entry.title,
                            'url': entry.link,
                            'score': 0,  # RSS feeds don't have scores
                            'source': feed.feed.get('title', category),
                            'category': category
                        })
                    else:
                        logger.debug(f"Filtered out: {entry.title[:50]}...")

            logger.info(f"✓ Fetched {len(stories)} stories from {category} RSS")

        except Exception as e:
            logger.warning(f"Failed to fetch RSS from {feed_url}: {e}")

        return stories

    def fetch_tech_news_from_sources(self) -> List[Dict]:
        """
        Fetches tech news from multiple curated sources.
        Includes AI, crypto, finance, and general tech.
        Returns list of {title, url, source, category} dicts.
        """
        stories = []

        # 1. Hacker News (most reliable for tech/dev content)
        hn_stories = self.fetch_hacker_news_top_stories()
        for story in hn_stories:
            story['category'] = 'tech'
        stories.extend(hn_stories)

        # 2. AI News RSS Feeds
        for feed_url in self.feeds['ai']:
            ai_stories = self.fetch_rss_feed(feed_url, 'ai', limit=5)
            stories.extend(ai_stories)

        # 3. Crypto News RSS Feeds
        for feed_url in self.feeds['crypto']:
            crypto_stories = self.fetch_rss_feed(feed_url, 'crypto', limit=5)
            stories.extend(crypto_stories)

        # 4. Finance News RSS Feeds
        for feed_url in self.feeds['finance']:
            finance_stories = self.fetch_rss_feed(feed_url, 'finance', limit=5)
            stories.extend(finance_stories)

        logger.info(f"✓ Total stories fetched: {len(stories)} across all sources")
        return stories

    def get_trending_story(self, preferred_categories: List[str] = None) -> Optional[Dict]:
        """
        Gets a single trending story suitable for posting.
        Prioritizes AI, then crypto, then finance, then general tech.

        Args:
            preferred_categories: List of categories to prefer ['ai', 'crypto', 'finance', 'tech']
        """
        stories = self.fetch_tech_news_from_sources()

        if not stories:
            logger.warning("No stories fetched from any source")
            return None

        # Default preference: AI > crypto > finance > tech
        if not preferred_categories:
            preferred_categories = ['ai', 'crypto', 'finance', 'tech']

        # Enhanced keywords with category-specific scoring
        keyword_weights = {
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'llm', 'neural',
                   'gpt', 'claude', 'gemini', 'openai', 'anthropic', 'deepmind', 'chatgpt',
                   'transformer', 'model', 'training', 'inference'],
            'crypto': ['crypto', 'bitcoin', 'ethereum', 'blockchain', 'defi', 'nft',
                      'web3', 'token', 'mining', 'wallet', 'exchange', 'coin'],
            'finance': ['stock', 'market', 'trading', 'investment', 'economy', 'fed',
                       'inflation', 'earnings', 'revenue', 'ipo', 'merger'],
            'tech': ['programming', 'developer', 'code', 'python', 'javascript',
                    'framework', 'library', 'open source', 'github', 'api',
                    'startup', 'google', 'microsoft', 'meta', 'apple']
        }

        # Score stories based on relevance and category preference
        scored_stories = []
        for story in stories:
            title_lower = story['title'].lower()
            category = story.get('category', 'tech')

            # Category preference bonus (AI gets highest)
            category_bonus = 0
            if category in preferred_categories:
                category_bonus = (len(preferred_categories) - preferred_categories.index(category)) * 100

            # Keyword matching score
            keyword_score = 0
            for cat, keywords in keyword_weights.items():
                matches = sum(1 for kw in keywords if kw in title_lower)
                if cat == category:
                    keyword_score += matches * 30  # Higher weight for category-specific keywords
                else:
                    keyword_score += matches * 10

            # HN score (if available)
            hn_score = story.get('score', 0) / 10

            # Combined score: category bonus + keyword score + HN score
            combined_score = category_bonus + keyword_score + hn_score

            scored_stories.append({
                **story,
                'relevance_score': combined_score
            })

        # Sort by relevance
        scored_stories.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Log top 10 for debugging (increased from 5 for more visibility)
        logger.info("Top 10 stories by score:")
        for i, story in enumerate(scored_stories[:10]):
            logger.info(f"  {i+1}. [{story['category'].upper()}] {story['title'][:40]}... (score: {story['relevance_score']:.1f}) from {story.get('source', 'unknown')}")

        # Pick from top 15 to add MORE variety (not just top 5)
        # This prevents Google blog from dominating
        top_stories = scored_stories[:15]
        if top_stories:
            selected = random.choice(top_stories)
            logger.info(f"✓ Selected: [{selected['category'].upper()}] {selected['title'][:60]}... (score: {selected['relevance_score']:.1f}) from {selected.get('source', 'unknown')}")
            return selected

        return None

    def get_multiple_stories(self, count: int = 5, preferred_categories: List[str] = None) -> List[Dict]:
        """
        Gets multiple trending stories for AI to choose from.
        Returns list of top stories (already scored and sorted).

        Args:
            count: Number of stories to return (default 5)
            preferred_categories: List of categories to prefer
        """
        stories = self.fetch_tech_news_from_sources()

        if not stories:
            return []

        # Default preference
        if not preferred_categories:
            preferred_categories = ['ai', 'crypto', 'finance', 'tech']

        # Keyword weights (same as get_trending_story)
        keyword_weights = {
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'llm', 'neural',
                   'gpt', 'claude', 'gemini', 'openai', 'anthropic', 'chatgpt'],
            'crypto': ['crypto', 'bitcoin', 'ethereum', 'blockchain', 'defi', 'web3'],
            'finance': ['stock', 'market', 'trading', 'investment', 'economy'],
            'tech': ['programming', 'developer', 'code', 'framework', 'startup']
        }

        # Score stories
        scored = []
        for story in stories:
            title_lower = story['title'].lower()
            category = story.get('category', 'tech')

            score = 0
            if category in preferred_categories:
                score += (len(preferred_categories) - preferred_categories.index(category)) * 100

            for cat, keywords in keyword_weights.items():
                matches = sum(1 for kw in keywords if kw in title_lower)
                score += matches * (30 if cat == category else 10)

            score += story.get('score', 0) / 10
            scored.append({**story, 'relevance_score': score})

        scored.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Return top N with variety (pick from different categories)
        result = []
        seen_categories = set()

        for story in scored:
            cat = story.get('category', 'tech')
            # Ensure category variety in results
            if len(result) < count:
                # Prioritize diversity in first few
                if cat not in seen_categories or len(result) >= count // 2:
                    result.append(story)
                    seen_categories.add(cat)

        logger.info(f"Returning {len(result)} stories for AI selection")
        return result[:count]

    def validate_url(self, url: str) -> bool:
        """Quick validation that URL is accessible."""
        try:
            response = self.session.head(url, timeout=5, allow_redirects=True)
            return response.status_code < 400
        except Exception:
            return False

    def fetch_article_content(self, url: str) -> str:
        """
        Fetches the article content from a URL.
        Returns the raw HTML content or empty string if failed.
        """
        try:
            logger.info(f"Fetching article content from: {url[:80]}...")
            response = self.session.get(url, timeout=10, allow_redirects=True)
            response.raise_for_status()

            # Return first 50KB of content (enough for context, not too much)
            content = response.text[:50000]
            logger.info(f"Fetched {len(content)} chars from article")
            return content

        except Exception as e:
            logger.warning(f"Failed to fetch article content: {e}")
            return ""
