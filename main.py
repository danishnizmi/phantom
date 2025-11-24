import os
import sys
import logging
import tweepy
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Config, get_secret
from brain import AgentBrain
from veo_client import VeoClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=60))
def post_tweet_v2(client, text, **kwargs):
    return client.create_tweet(text=text, **kwargs)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=60))
def upload_media_v1(api, filename, **kwargs):
    return api.media_upload(filename, **kwargs)

def get_twitter_api():
    """Authenticates with X (Twitter) API."""
    required_secrets = [
        "TWITTER_CONSUMER_KEY",
        "TWITTER_CONSUMER_SECRET",
        "TWITTER_ACCESS_TOKEN",
        "TWITTER_ACCESS_TOKEN_SECRET"
    ]
    
    secrets = {}
    for secret_id in required_secrets:
        val = get_secret(secret_id)
        if not val:
            raise ValueError(f"Missing required secret: {secret_id}")
        secrets[secret_id] = val

    auth = tweepy.OAuth1UserHandler(
        secrets["TWITTER_CONSUMER_KEY"],
        secrets["TWITTER_CONSUMER_SECRET"],
        secrets["TWITTER_ACCESS_TOKEN"],
        secrets["TWITTER_ACCESS_TOKEN_SECRET"]
    )
    
    # Create clients
    api_v1 = tweepy.API(auth)
    client_v2 = tweepy.Client(
        consumer_key=secrets["TWITTER_CONSUMER_KEY"],
        consumer_secret=secrets["TWITTER_CONSUMER_SECRET"],
        access_token=secrets["TWITTER_ACCESS_TOKEN"],
        access_token_secret=secrets["TWITTER_ACCESS_TOKEN_SECRET"]
    )
    
    # Verify credentials
    try:
        api_v1.verify_credentials()
    except Exception as e:
        raise ValueError(f"Twitter authentication failed: {e}")
        
    return api_v1, client_v2

def main():
    logger.info("Starting Tech Influencer Agent...")
    
    # 1. Validate Environment & Secrets FIRST (Fail Fast)
    try:
        Config.validate()
        api_v1, client_v2 = get_twitter_api()
    except Exception as e:
        logger.critical(f"Initialization Error: {e}")
        sys.exit(1) # Exit with error code

    # 2. Initialize Brain
    try:
        brain = AgentBrain()
    except Exception as e:
        logger.critical(f"Failed to initialize Brain: {e}")
        sys.exit(1)

    # 3. Get Strategy
    try:
        strategy = brain.get_strategy()
        logger.info(f"Strategy decided: {strategy}")
    except Exception as e:
        logger.error(f"Failed to generate strategy: {e}")
        sys.exit(1)

    # 4. Execute Strategy
    try:
        if strategy["type"] == "video":
            video_path = None
            try:
                veo = VeoClient(project_id=Config.PROJECT_ID, region=Config.REGION)
                video_path = veo.generate_video(strategy["video_prompt"])
                
                # Upload Video (requires v1.1 API)
                media = upload_media_v1(api_v1, video_path, chunked=True, media_category="tweet_video")
                
                # Post Tweet with Video (requires v2 API)
                post_tweet_v2(client_v2, text=strategy["content"], media_ids=[media.media_id])
                logger.info("Video posted successfully!")
                brain.log_post(strategy, success=True)
                
            except Exception as e:
                logger.error(f"Video generation or upload failed: {e}")
                logger.info("Falling back to text with URL...")

                # Fallback logic - post text with URL (if available)
                try:
                    caption = strategy['content']
                    source_url = strategy.get('source_url')

                    # If we have a URL, create proper fallback with citation
                    if source_url:
                        fallback_text = f"{caption}\n\n{source_url}"
                        # Ensure under 280 chars
                        if len(fallback_text) > 280:
                            max_caption = 280 - len(source_url) - 4  # -4 for \n\n spacing
                            fallback_text = f"{caption[:max_caption]}...\n\n{source_url}"
                    else:
                        fallback_text = caption

                    post_tweet_v2(client_v2, text=fallback_text)
                    logger.info("Posted text with URL after video failure")
                    brain.log_post(strategy, success=True, error=f"Video failed but posted text with URL. Error: {e}")
                except Exception as fallback_error:
                    logger.error(f"Fallback tweet also failed: {fallback_error}")
                    brain.log_post(strategy, success=False, error=f"Video and Fallback failed. Error: {e} | Fallback: {fallback_error}")
                    sys.exit(1) # Fail job if even fallback fails

            finally:
                # Cleanup video file
                if video_path and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                        logger.info(f"Cleaned up video file: {video_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup video file: {cleanup_error}")

        elif strategy["type"] == "image":
            image_path = None
            try:
                image_path = brain.generate_image(strategy["image_prompt"])
                
                # Upload Image
                media = upload_media_v1(api_v1, image_path)
                
                # Post Tweet with Image
                post_tweet_v2(client_v2, text=strategy["content"], media_ids=[media.media_id])
                logger.info("Image posted successfully!")
                brain.log_post(strategy, success=True)
                
            except Exception as e:
                logger.error(f"Image generation or upload failed: {e}")
                logger.info("Falling back to text with URL...")

                # Fallback to text with URL
                try:
                    caption = strategy['content']
                    source_url = strategy.get('source_url')

                    # If we have a URL, create proper fallback with citation
                    if source_url:
                        fallback_text = f"{caption}\n\n{source_url}"
                        # Ensure under 280 chars
                        if len(fallback_text) > 280:
                            max_caption = 280 - len(source_url) - 4  # -4 for \n\n spacing
                            fallback_text = f"{caption[:max_caption]}...\n\n{source_url}"
                    else:
                        fallback_text = caption

                    post_tweet_v2(client_v2, text=fallback_text)
                    logger.info("Posted text with URL after image failure")
                    brain.log_post(strategy, success=True, error=f"Image failed but posted text with URL. Error: {e}")
                except Exception as fallback_error:
                    logger.error(f"Fallback tweet also failed: {fallback_error}")
                    brain.log_post(strategy, success=False, error=f"Image and Fallback failed. Error: {e}")
                    sys.exit(1)
            finally:
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception:
                        pass

        elif strategy["type"] in ["thread", "text"]:
            tweets = strategy["content"]
            # Ensure it's a list
            if isinstance(tweets, str):
                tweets = [tweets]
                
            previous_tweet_id = None
            posted_tweets = []
            
            try:
                for tweet_text in tweets:
                    # Basic length check
                    if len(tweet_text) > 280:
                        logger.warning(f"Tweet too long, truncating: {tweet_text[:50]}...")
                        tweet_text = tweet_text[:277] + "..."
                    
                    if previous_tweet_id:
                        response = post_tweet_v2(client_v2, text=tweet_text, in_reply_to_tweet_id=previous_tweet_id)
                    else:
                        response = post_tweet_v2(client_v2, text=tweet_text)
                    
                    previous_tweet_id = response.data['id']
                    posted_tweets.append(previous_tweet_id)
                
                logger.info(f"Post successful! IDs: {posted_tweets}")
                brain.log_post(strategy, success=True)
            except Exception as e:
                logger.error(f"Failed to post text: {e}")
                brain.log_post(strategy, success=False, error=f"Partial failure. Posted: {len(posted_tweets)}. Error: {e}")
                sys.exit(1)

    except Exception as e:
        logger.critical(f"Critical execution error: {e}")
        # Try to log to Firestore if possible
        try:
             brain.log_post(strategy, success=False, error=f"Critical Error: {e}")
        except:
             pass
        sys.exit(1)

if __name__ == "__main__":
    main()
