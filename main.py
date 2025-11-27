import os
import sys
import logging
import tweepy
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Config, get_secret

# Lazy imports for cold start optimization - only import heavy modules when needed
# from brain import AgentBrain  # Moved to after scheduler check
# from veo_client import VeoClient  # Imported when needed

# Check for scheduling mode
FORCE_POST = os.getenv("FORCE_POST", "false").lower() == "true"

# Configure logging - will be enhanced with structured logging below
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Try to enable structured logging for GCP Cloud Logging
try:
    import google.cloud.logging
    from google.cloud.logging.handlers import StructuredLogHandler

    # Setup structured logging for GCP
    client = google.cloud.logging.Client()
    client.setup_logging()
    logger.info("Structured logging enabled for GCP Cloud Logging")
except ImportError:
    logger.debug("google-cloud-logging not available, using standard logging")
except Exception as e:
    logger.warning(f"Could not setup GCP structured logging: {e}")


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

    # COLD START OPTIMIZATION: Check scheduler FIRST before heavy initialization
    # This avoids wasting API calls and compute time if we're not going to post
    if not FORCE_POST:
        from scheduler import should_post_lightweight
        should_post, reason = should_post_lightweight()
        if not should_post:
            logger.info(f"Skipping post: {reason}")
            logger.info("Set FORCE_POST=true to override scheduler")
            logger.info("Cold start optimization: No heavy initialization performed")
            sys.exit(0)  # Clean exit - not an error, minimal cost
        logger.info(f"Scheduler approved: {reason}")
    else:
        logger.info("FORCE_POST enabled, bypassing scheduler check")

    # 1. Validate Environment & Secrets (only after scheduler approves)
    try:
        Config.validate()
        api_v1, client_v2 = get_twitter_api()
    except Exception as e:
        logger.critical(f"Initialization Error: {e}")
        sys.exit(1)

    # 2. Initialize Brain (LAZY - only after we know we'll post)
    # This is the expensive part: Vertex AI init, model discovery, Firestore
    logger.info("Initializing AgentBrain (heavy initialization)...")
    try:
        from brain import AgentBrain
        brain = AgentBrain()
    except Exception as e:
        logger.critical(f"Failed to initialize Brain: {e}")
        sys.exit(1)

    # 3. Get Strategy (may return None if no quality content available)
    try:
        strategy = brain.get_strategy()

        if strategy is None:
            logger.info("No quality content available - skipping this post cycle")
            logger.info("This is normal - we only post when we have good content")
            sys.exit(0)  # Clean exit, not an error

        logger.info(f"Strategy decided: {strategy}")
    except Exception as e:
        logger.error(f"Failed to generate strategy: {e}")
        sys.exit(1)

    # 4. Execute Strategy
    try:
        if strategy["type"] == "video":
            video_path = None
            try:
                # Lazy import VeoClient only when needed
                from veo_client import VeoClient
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

        elif strategy["type"] == "infographic":
            # Handle infographic posts (educational content)
            image_path = None
            try:
                # Use infographic generator with topic and key points
                topic = strategy.get("topic", "Tech Trends")
                key_points = strategy.get("key_points", [])
                source_url = strategy.get("source_url")
                image_prompt = strategy.get("image_prompt")

                if brain.infographic_generator and not image_prompt:
                    # Use dedicated infographic generator
                    infographic_result = brain.generate_infographic(
                        topic=topic,
                        key_points=key_points,
                        source_url=source_url
                    )
                    image_path = infographic_result.get('image_path')
                    # Update caption if infographic generator provided one
                    if infographic_result.get('content'):
                        strategy['content'] = infographic_result['content']
                else:
                    # Use the image_prompt from strategy (generated by brain.get_strategy)
                    # This uses the regular Imagen generator with infographic-specific prompt
                    logger.info("Using Imagen with infographic prompt from strategy")
                    prompt = image_prompt or f"Professional tech infographic about {topic}. Clean diagram, data visualization, educational content."
                    image_path = brain.generate_image(prompt)

                # Upload Image
                media = upload_media_v1(api_v1, image_path)

                # Post Tweet with Infographic
                post_tweet_v2(client_v2, text=strategy["content"], media_ids=[media.media_id])
                logger.info("Infographic posted successfully!")
                brain.log_post(strategy, success=True)

            except Exception as e:
                logger.error(f"Infographic generation or upload failed: {e}")
                logger.info("Falling back to text with URL...")

                # Fallback to text with URL
                try:
                    caption = strategy.get('content', strategy.get('topic', 'Tech update'))
                    source_url = strategy.get('source_url')

                    if source_url:
                        fallback_text = f"{caption}\n\n{source_url}"
                        if len(fallback_text) > 280:
                            max_caption = 280 - len(source_url) - 4
                            fallback_text = f"{caption[:max_caption]}...\n\n{source_url}"
                    else:
                        fallback_text = caption if len(caption) <= 280 else caption[:277] + "..."

                    post_tweet_v2(client_v2, text=fallback_text)
                    logger.info("Posted text with URL after infographic failure")
                    brain.log_post(strategy, success=True, error=f"Infographic failed but posted text with URL. Error: {e}")
                except Exception as fallback_error:
                    logger.error(f"Fallback tweet also failed: {fallback_error}")
                    brain.log_post(strategy, success=False, error=f"Infographic and Fallback failed. Error: {e}")
                    sys.exit(1)
            finally:
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception:
                        pass

        elif strategy["type"] == "meme":
            # MEME: Use fetched meme from Reddit (meme_local_path) or fallback to text
            image_path = strategy.get("meme_local_path")  # Pre-downloaded meme
            try:
                if image_path and os.path.exists(image_path):
                    # We have a fetched meme - upload and post
                    logger.info(f"Posting fetched meme from: {strategy.get('meme_source', 'Reddit')}")
                    file_size = os.path.getsize(image_path)
                    logger.info(f"Uploading meme file: {image_path} ({file_size} bytes)")

                    # Upload media - use chunked upload for GIFs
                    is_gif = image_path.lower().endswith('.gif')
                    if is_gif:
                        logger.info("Using chunked upload for GIF...")
                        media = upload_media_v1(api_v1, image_path, chunked=True, media_category="tweet_gif")
                    else:
                        media = upload_media_v1(api_v1, image_path)

                    logger.info(f"Media upload complete. media_id: {media.media_id}, media_id_string: {getattr(media, 'media_id_string', 'N/A')}")

                    # Post tweet with media
                    response = post_tweet_v2(client_v2, text=strategy["content"], media_ids=[media.media_id])
                    logger.info(f"Tweet posted! Response: {response.data if hasattr(response, 'data') else response}")
                    logger.info(f"Meme posted successfully! Source: {strategy.get('meme_title', '')[:50]}")
                    brain.log_post(strategy, success=True)
                else:
                    # No meme image - post as text (brain.py already set content)
                    logger.info("No meme image, posting as text")
                    content = strategy["content"]
                    if isinstance(content, list):
                        content = content[0]
                    post_tweet_v2(client_v2, text=content)
                    logger.info("Posted meme-style text successfully!")
                    brain.log_post(strategy, success=True)

            except Exception as e:
                logger.error(f"Meme posting failed: {e}")
                # Fallback to text
                try:
                    content = strategy["content"]
                    if isinstance(content, list):
                        content = content[0]
                    post_tweet_v2(client_v2, text=content)
                    logger.info("Posted text fallback after meme failure")
                    brain.log_post(strategy, success=True, error=f"Meme failed, posted text. Error: {e}")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    brain.log_post(strategy, success=False, error=str(e))
                    sys.exit(1)
            finally:
                # Cleanup downloaded meme
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        logger.info(f"Cleaned up meme file: {image_path}")
                    except Exception as cleanup_err:
                        logger.warning(f"Failed to cleanup meme file: {cleanup_err}")

        elif strategy["type"] == "image":
            # IMAGE: AI-generated image with Imagen
            image_path = None
            try:
                image_prompt = strategy.get("image_prompt")
                if not image_prompt:
                    raise ValueError("Missing image_prompt for image post")

                image_path = brain.generate_image(image_prompt)

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

                    if source_url:
                        fallback_text = f"{caption}\n\n{source_url}"
                        if len(fallback_text) > 280:
                            max_caption = 280 - len(source_url) - 4
                            fallback_text = f"{caption[:max_caption]}...\n\n{source_url}"
                    else:
                        fallback_text = caption

                    post_tweet_v2(client_v2, text=fallback_text)
                    logger.info("Posted text with URL after image failure")
                    brain.log_post(strategy, success=True, error=f"Image failed but posted text. Error: {e}")
                except Exception as fallback_error:
                    logger.error(f"Fallback tweet also failed: {fallback_error}")
                    brain.log_post(strategy, success=False, error=str(e))
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
        except Exception as log_err:
            logger.warning(f"Failed to log error to Firestore: {log_err}")
        sys.exit(1)

if __name__ == "__main__":
    main()
