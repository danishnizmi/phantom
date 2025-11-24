import logging
import requests
from typing import Dict, List, Optional
import random

logger = logging.getLogger(__name__)

class NewsFetcher:
    """Fetches real tech news from various sources."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; TechInfluencerBot/1.0)'
        })

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

    def fetch_tech_news_from_sources(self) -> List[Dict]:
        """
        Fetches tech news from multiple curated sources.
        Returns list of {title, url, source} dicts.
        """
        stories = []

        # Try Hacker News first (most reliable)
        hn_stories = self.fetch_hacker_news_top_stories()
        stories.extend(hn_stories)

        # Could add more sources here:
        # - TechCrunch RSS
        # - The Verge RSS
        # - Ars Technica RSS

        return stories

    def get_trending_story(self) -> Optional[Dict]:
        """
        Gets a single trending tech story suitable for posting.
        Prioritizes high-scoring stories and filters for interesting topics.
        """
        stories = self.fetch_tech_news_from_sources()

        if not stories:
            logger.warning("No stories fetched from any source")
            return None

        # Filter for tech-relevant keywords
        tech_keywords = [
            'ai', 'ml', 'llm', 'gpt', 'claude', 'gemini', 'openai',
            'programming', 'developer', 'code', 'python', 'javascript',
            'framework', 'library', 'open source', 'github',
            'startup', 'google', 'microsoft', 'meta', 'apple',
            'api', 'database', 'cloud', 'aws', 'gcp', 'azure',
            'security', 'privacy', 'crypto', 'blockchain',
            'web', 'mobile', 'app', 'software', 'hardware'
        ]

        # Score stories based on relevance and popularity
        scored_stories = []
        for story in stories:
            title_lower = story['title'].lower()

            # Calculate relevance score
            keyword_matches = sum(1 for kw in tech_keywords if kw in title_lower)
            hn_score = story.get('score', 0)

            # Combined score: keyword relevance + HN score
            combined_score = (keyword_matches * 50) + (hn_score / 10)

            scored_stories.append({
                **story,
                'relevance_score': combined_score
            })

        # Sort by relevance
        scored_stories.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Pick from top 5 to add some variety
        top_stories = scored_stories[:5]
        if top_stories:
            selected = random.choice(top_stories)
            logger.info(f"Selected story: {selected['title'][:60]}... (score: {selected['relevance_score']:.1f})")
            return selected

        return None

    def validate_url(self, url: str) -> bool:
        """Quick validation that URL is accessible."""
        try:
            response = self.session.head(url, timeout=5, allow_redirects=True)
            return response.status_code < 400
        except Exception:
            return False
