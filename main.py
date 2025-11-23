import os
import tweepy
from config import Config, get_secret
from brain import AgentBrain
from veo_client import VeoClient

def get_twitter_api():
    """Authenticates with X (Twitter) API."""
    consumer_key = get_secret("TWITTER_CONSUMER_KEY")
    consumer_secret = get_secret("TWITTER_CONSUMER_SECRET")
    access_token = get_secret("TWITTER_ACCESS_TOKEN")
    access_token_secret = get_secret("TWITTER_ACCESS_TOKEN_SECRET")

    auth = tweepy.OAuth1UserHandler(
        consumer_key, consumer_secret, access_token, access_token_secret
    )
    return tweepy.API(auth), tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret
    )

def main():
    print("Starting Tech Influencer Agent...")
    
    # Initialize components
    try:
        brain = AgentBrain()
        # Only init Veo if we might need it (optimization)
        # But for simplicity, we can init it or lazy load it.
        # We'll lazy load inside the loop if needed.
    except Exception as e:
        print(f"Failed to initialize Brain: {e}")
        return

    # Get Strategy
    try:
        strategy = brain.get_strategy()
        print(f"Strategy decided: {strategy}")
    except Exception as e:
        print(f"Failed to generate strategy: {e}")
        return

    # Execute Strategy
    try:
        api_v1, client_v2 = get_twitter_api()
        
        if strategy["type"] == "video":
            try:
                veo = VeoClient(project_id=Config.PROJECT_ID, region=Config.REGION)
                video_path = veo.generate_video(strategy["video_prompt"])
                
                # Upload Video (requires v1.1 API)
                media = api_v1.media_upload(video_path, chunked=True, media_category="tweet_video")
                
                # Post Tweet with Video (requires v2 API)
                client_v2.create_tweet(text=strategy["content"], media_ids=[media.media_id])
                print("Video posted successfully!")
                brain.log_post(strategy, success=True)
                
            except Exception as e:
                print(f"Video generation or upload failed: {e}")
                print("Falling back to text thread...")
                # Fallback logic: Convert to thread or just post the caption
                fallback_text = f"{strategy['content']} (Video generation failed, but the tech is still cool!)"
                client_v2.create_tweet(text=fallback_text)
                brain.log_post(strategy, success=False, error=str(e))

        elif strategy["type"] == "thread":
            tweets = strategy["content"]
            previous_tweet_id = None
            
            for tweet_text in tweets:
                if previous_tweet_id:
                    response = client_v2.create_tweet(text=tweet_text, in_reply_to_tweet_id=previous_tweet_id)
                else:
                    response = client_v2.create_tweet(text=tweet_text)
                previous_tweet_id = response.data['id']
            
            print("Thread posted successfully!")
            brain.log_post(strategy, success=True)

    except Exception as e:
        print(f"Critical execution error: {e}")
        brain.log_post(strategy, success=False, error=str(e))

if __name__ == "__main__":
    main()
