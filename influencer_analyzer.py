"""
Influencer Analyzer - Fetches trending content from top accounts in each category.
Uses Twitter API to discover what's popular and analyzes posting patterns.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import tweepy

logger = logging.getLogger(__name__)

# Import secret fetcher
try:
    from config import get_secret
    SECRET_MANAGER_AVAILABLE = True
except ImportError:
    SECRET_MANAGER_AVAILABLE = False


class InfluencerAnalyzer:
    """
    Analyzes trending content from top influencers in tech categories.
    Uses Twitter search API to find high-engagement posts dynamically.
    """

    # Search queries for each category (free tier compatible - no min_faves)
    CATEGORY_QUERIES = {
        'ai': [
            'AI announcement -is:retweet lang:en',
            'GPT OR Claude OR Gemini -is:retweet lang:en',
            'machine learning -is:retweet lang:en',
            'LLM release -is:retweet lang:en',
        ],
        'crypto': [
            'Bitcoin OR Ethereum -is:retweet lang:en',
            'crypto news -is:retweet lang:en',
            'DeFi OR NFT -is:retweet lang:en',
            'blockchain -is:retweet lang:en',
        ],
        'tech': [
            'tech startup -is:retweet lang:en',
            'silicon valley -is:retweet lang:en',
            'developer tools -is:retweet lang:en',
            'open source -is:retweet lang:en',
        ],
        'finance': [
            'stock market -is:retweet lang:en',
            'Fed OR inflation -is:retweet lang:en',
            'fintech -is:retweet lang:en',
        ]
    }

    # Regions to exclude (ISO country codes for filtering)
    EXCLUDED_REGIONS = {'IN'}  # India

    # Free tier: 1 search request per 15 minutes
    FREE_TIER_COOLDOWN_SECONDS = 15 * 60  # 15 minutes

    def __init__(self, bearer_token: str = None):
        """Initialize with Twitter API credentials from Secret Manager."""
        self.bearer_token = bearer_token
        self.client = None
        self._last_api_call = None  # Track last API call time
        self._cached_results = {}   # Cache results to avoid repeated calls

        # Try to load from Secret Manager if not provided
        if not self.bearer_token and SECRET_MANAGER_AVAILABLE:
            try:
                self.bearer_token = get_secret('TWITTER_BEARER_TOKEN')
                logger.info("Loaded Twitter bearer token from Secret Manager")
            except Exception as e:
                logger.warning(f"Could not load Twitter bearer token: {e}")

        if self.bearer_token:
            try:
                self.client = tweepy.Client(bearer_token=self.bearer_token)
                logger.info("✓ Influencer analyzer initialized with Twitter API (FREE TIER - 1 req/15min)")
            except Exception as e:
                logger.warning(f"Could not initialize Twitter client: {e}")
        else:
            logger.warning("Twitter bearer token not available - influencer analysis disabled")

    def _can_make_api_call(self) -> bool:
        """Check if we can make an API call (free tier: 1 per 15 min)."""
        if self._last_api_call is None:
            return True

        elapsed = (datetime.now() - self._last_api_call).total_seconds()
        if elapsed < self.FREE_TIER_COOLDOWN_SECONDS:
            remaining = int(self.FREE_TIER_COOLDOWN_SECONDS - elapsed)
            logger.info(f"Twitter API rate limit: {remaining}s remaining until next call allowed")
            return False
        return True

    def get_rate_limit_status(self) -> Dict:
        """Returns current rate limit status for monitoring."""
        if self._last_api_call is None:
            return {
                'can_call': True,
                'last_call': None,
                'seconds_remaining': 0,
                'cached_categories': list(self._cached_results.keys())
            }

        elapsed = (datetime.now() - self._last_api_call).total_seconds()
        remaining = max(0, self.FREE_TIER_COOLDOWN_SECONDS - elapsed)

        return {
            'can_call': elapsed >= self.FREE_TIER_COOLDOWN_SECONDS,
            'last_call': self._last_api_call.isoformat(),
            'seconds_remaining': int(remaining),
            'cached_categories': list(self._cached_results.keys())
        }

    def _is_excluded_region(self, user_data: dict) -> bool:
        """
        Check if user appears to be from an excluded region.
        Uses location field and bio analysis.
        """
        if not user_data:
            return False

        location = (user_data.get('location') or '').lower()
        bio = (user_data.get('description') or '').lower()
        name = (user_data.get('name') or '').lower()

        # Location-based filtering
        excluded_locations = [
            'india', 'mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai',
            'kolkata', 'pune', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur',
            'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri',
            'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik',
            'faridabad', 'meerut', 'rajkot', 'varanasi', 'srinagar', 'noida',
            'gurgaon', 'coimbatore', 'madurai', 'kochi', 'chandigarh',
            '🇮🇳', 'bharat', 'hindustan',
        ]

        for loc in excluded_locations:
            if loc in location or loc in bio:
                return True

        return False

    def fetch_trending_posts(self, category: str, limit: int = 10) -> List[Dict]:
        """
        Fetches high-engagement posts for a category using search.
        Returns list of posts with engagement metrics.
        Free tier: 1 request per 15 minutes - uses caching.
        """
        if not self.client:
            logger.warning("Twitter client not available")
            return []

        # Check cache first (valid for 15 min cooldown period)
        cache_key = f"{category}_{limit}"
        if cache_key in self._cached_results:
            cached_time, cached_data = self._cached_results[cache_key]
            cache_age = (datetime.now() - cached_time).total_seconds()
            if cache_age < self.FREE_TIER_COOLDOWN_SECONDS:
                logger.info(f"Using cached results for {category} ({int(cache_age)}s old, {int(self.FREE_TIER_COOLDOWN_SECONDS - cache_age)}s until refresh)")
                return cached_data

        # Check rate limit before making API call
        if not self._can_make_api_call():
            logger.info(f"Rate limited - returning cached/empty results for {category}")
            # Return stale cache if available, otherwise empty
            if cache_key in self._cached_results:
                return self._cached_results[cache_key][1]
            return []

        queries = self.CATEGORY_QUERIES.get(category, self.CATEGORY_QUERIES['tech'])
        all_posts = []

        # Free tier: only use first query (1 request per 15 min)
        queries_to_use = queries[:1]

        for query in queries_to_use:
            try:
                # Search recent tweets
                tweets = self.client.search_recent_tweets(
                    query=query,
                    max_results=10,  # Reduced for free tier
                    tweet_fields=['created_at', 'public_metrics', 'author_id', 'lang'],
                    user_fields=['name', 'username', 'location', 'description', 'public_metrics'],
                    expansions=['author_id']
                )

                # Track API call time for rate limiting
                self._last_api_call = datetime.now()
                logger.info(f"Twitter API call made at {self._last_api_call.strftime('%H:%M:%S')} - next allowed in 15 min")

                if not tweets.data:
                    continue

                # Build user lookup
                users = {}
                if tweets.includes and 'users' in tweets.includes:
                    for user in tweets.includes['users']:
                        users[user.id] = {
                            'name': user.name,
                            'username': user.username,
                            'location': getattr(user, 'location', ''),
                            'description': getattr(user, 'description', ''),
                            'followers': user.public_metrics.get('followers_count', 0) if hasattr(user, 'public_metrics') else 0
                        }

                for tweet in tweets.data:
                    # Skip non-English tweets
                    if tweet.lang and tweet.lang != 'en':
                        continue

                    # Get author info
                    author = users.get(tweet.author_id, {})

                    # Skip if from excluded region
                    if self._is_excluded_region(author):
                        logger.debug(f"Skipping tweet from excluded region: {author.get('username')}")
                        continue

                    metrics = tweet.public_metrics or {}
                    engagement = (
                        metrics.get('like_count', 0) +
                        metrics.get('retweet_count', 0) * 2 +
                        metrics.get('reply_count', 0) * 3
                    )

                    all_posts.append({
                        'text': tweet.text,
                        'author': author.get('username', 'unknown'),
                        'author_name': author.get('name', ''),
                        'followers': author.get('followers', 0),
                        'likes': metrics.get('like_count', 0),
                        'retweets': metrics.get('retweet_count', 0),
                        'replies': metrics.get('reply_count', 0),
                        'engagement_score': engagement,
                        'category': category,
                        'created_at': str(tweet.created_at) if tweet.created_at else None
                    })

            except Exception as e:
                logger.warning(f"Error fetching tweets for query '{query[:30]}...': {e}")
                continue

        # Sort by engagement and return top posts
        all_posts.sort(key=lambda x: x['engagement_score'], reverse=True)
        result = all_posts[:limit]

        # Cache results for rate limiting period
        self._cached_results[cache_key] = (datetime.now(), result)
        logger.info(f"Found {len(result)} trending posts for {category} (cached for 15 min)")

        return result

    def get_trending_topics(self, categories: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Gets trending posts across multiple categories.
        Returns dict mapping category -> list of top posts.

        Note: Free tier only allows 1 API call per 15 min, so this
        will return cached results for most categories.
        """
        if categories is None:
            categories = ['ai', 'crypto', 'tech', 'finance']

        trending = {}
        api_calls_made = 0

        for category in categories:
            # Free tier: only make 1 fresh API call per invocation
            # Rest will use cache or return empty
            if api_calls_made > 0 and not self._can_make_api_call():
                logger.debug(f"Skipping fresh fetch for {category} - rate limited")

            posts = self.fetch_trending_posts(category, limit=5)
            if posts:
                trending[category] = posts
                # Check if this was a fresh call (not cached)
                cache_key = f"{category}_5"
                if cache_key in self._cached_results:
                    cache_time, _ = self._cached_results[cache_key]
                    if (datetime.now() - cache_time).total_seconds() < 5:
                        api_calls_made += 1

        logger.info(f"Trending topics: {len(trending)} categories with data")
        return trending

    def analyze_posting_style(self, posts: List[Dict]) -> Dict:
        """
        Analyzes common patterns in successful posts.
        Returns style insights for content generation.
        """
        if not posts:
            return {}

        # Analyze patterns
        avg_length = sum(len(p['text']) for p in posts) / len(posts)
        has_emoji = sum(1 for p in posts if any(ord(c) > 127 for c in p['text'])) / len(posts)
        has_link = sum(1 for p in posts if 'http' in p['text']) / len(posts)
        has_hashtag = sum(1 for p in posts if '#' in p['text']) / len(posts)
        has_mention = sum(1 for p in posts if '@' in p['text']) / len(posts)

        # Extract common words (simple approach)
        all_words = ' '.join(p['text'] for p in posts).lower().split()
        word_freq = {}
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'this', 'that', 'these', 'those', 'it', 'its', 'and', 'or',
                     'but', 'if', 'as', 'so', 'just', 'about', 'into', 'than'}
        for word in all_words:
            word = ''.join(c for c in word if c.isalnum())
            if word and len(word) > 3 and word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1

        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'avg_length': int(avg_length),
            'emoji_usage': round(has_emoji, 2),
            'link_usage': round(has_link, 2),
            'hashtag_usage': round(has_hashtag, 2),
            'mention_usage': round(has_mention, 2),
            'trending_words': [w[0] for w in top_words],
            'sample_count': len(posts),
            'avg_engagement': sum(p['engagement_score'] for p in posts) / len(posts)
        }

    def get_content_recommendations(self, category: str = 'ai') -> Dict:
        """
        Gets content recommendations based on trending analysis.
        Returns insights for strategy generation.
        """
        posts = self.fetch_trending_posts(category, limit=20)

        if not posts:
            return {
                'category': category,
                'has_data': False,
                'recommendation': 'No trending data available'
            }

        style = self.analyze_posting_style(posts)

        # Get top performing post as reference
        top_post = posts[0] if posts else None

        return {
            'category': category,
            'has_data': True,
            'style_insights': style,
            'top_post': {
                'text': top_post['text'][:200] + '...' if len(top_post['text']) > 200 else top_post['text'],
                'author': top_post['author'],
                'engagement': top_post['engagement_score']
            } if top_post else None,
            'trending_topics': style.get('trending_words', []),
            'recommendation': self._generate_recommendation(style)
        }

    def _generate_recommendation(self, style: Dict) -> str:
        """Generate posting recommendation based on style analysis."""
        recs = []

        if style.get('avg_length', 0) < 150:
            recs.append("Keep posts concise (under 150 chars)")
        elif style.get('avg_length', 0) > 250:
            recs.append("Longer, detailed posts perform well")

        if style.get('emoji_usage', 0) > 0.5:
            recs.append("Use emojis for engagement")

        if style.get('link_usage', 0) > 0.5:
            recs.append("Include source links")

        if style.get('hashtag_usage', 0) > 0.3:
            recs.append("Use relevant hashtags")

        if not recs:
            recs.append("Focus on timely, newsworthy content")

        return "; ".join(recs)
