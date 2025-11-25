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

    # Search queries for each category (dynamic discovery, no hardcoded accounts)
    CATEGORY_QUERIES = {
        'ai': [
            'AI announcement min_faves:1000',
            'GPT OR Claude OR Gemini min_faves:500 -is:retweet',
            'machine learning breakthrough min_faves:300',
            'LLM release min_faves:500',
        ],
        'crypto': [
            'Bitcoin OR Ethereum min_faves:1000 -is:retweet',
            'crypto news min_faves:500',
            'DeFi OR NFT announcement min_faves:300',
            'blockchain update min_faves:500',
        ],
        'tech': [
            'tech startup funding min_faves:500',
            'silicon valley news min_faves:300',
            'developer tools launch min_faves:300',
            'open source release min_faves:500',
        ],
        'finance': [
            'stock market min_faves:500 -is:retweet',
            'Fed OR inflation news min_faves:300',
            'fintech announcement min_faves:300',
        ]
    }

    # Regions to exclude (ISO country codes for filtering)
    EXCLUDED_REGIONS = {'IN'}  # India

    def __init__(self, bearer_token: str = None):
        """Initialize with Twitter API credentials from Secret Manager."""
        self.bearer_token = bearer_token
        self.client = None

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
                logger.info("✓ Influencer analyzer initialized with Twitter API")
            except Exception as e:
                logger.warning(f"Could not initialize Twitter client: {e}")
        else:
            logger.warning("Twitter bearer token not available - influencer analysis disabled")

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
        """
        if not self.client:
            logger.warning("Twitter client not available")
            return []

        queries = self.CATEGORY_QUERIES.get(category, self.CATEGORY_QUERIES['tech'])
        all_posts = []

        for query in queries:
            try:
                # Search recent tweets with high engagement
                tweets = self.client.search_recent_tweets(
                    query=query,
                    max_results=20,
                    tweet_fields=['created_at', 'public_metrics', 'author_id', 'lang'],
                    user_fields=['name', 'username', 'location', 'description', 'public_metrics'],
                    expansions=['author_id']
                )

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
        logger.info(f"Found {len(all_posts)} trending posts for {category}")

        return all_posts[:limit]

    def get_trending_topics(self, categories: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Gets trending posts across multiple categories.
        Returns dict mapping category -> list of top posts.
        """
        if categories is None:
            categories = ['ai', 'crypto', 'tech', 'finance']

        trending = {}
        for category in categories:
            posts = self.fetch_trending_posts(category, limit=5)
            if posts:
                trending[category] = posts

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
