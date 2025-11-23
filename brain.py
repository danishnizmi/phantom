import logging
import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import firestore
from config import Config
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentBrain:
    def __init__(self):
        self.project_id = Config.PROJECT_ID
        self.location = Config.REGION
        
        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel("gemini-1.5-flash-001")
        self.db = firestore.Client(project=self.project_id)
        self.collection = self.db.collection(Config.COLLECTION_NAME)

    def _get_trending_topic(self) -> str:
        """
        Asks Gemini to identify a trending tech topic.
        """
        prompt = "Identify a single, specific, currently trending technology topic or news item suitable for a tech influencer tweet. Return ONLY the topic name."
        try:
            response = self.model.generate_content(prompt)
            if not response.text:
                raise ValueError("Empty response from Gemini")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to get trending topic: {e}")
            return "Artificial Intelligence" # Safe fallback

    def _check_history(self, topic: str) -> bool:
        """
        Checks Firestore to see if we've recently posted about this topic.
        Returns True if we should SKIP this topic (duplicate), False otherwise.
        """
        try:
            # Check last 20 posts
            # Note: Requires composite index if using multiple fields, but here we just order by timestamp.
            # If collection is empty, this returns empty generator, which is fine.
            docs = self.collection.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20).stream()
            
            for doc in docs:
                data = doc.to_dict()
                # Improved matching: Check containment or exact match
                stored_topic = data.get("topic", "").lower()
                current_topic = topic.lower()
                
                if stored_topic == current_topic or current_topic in stored_topic or stored_topic in current_topic:
                    logger.info(f"Topic '{topic}' matches recent post '{stored_topic}'. Skipping.")
                    return True
            return False
        except Exception as e:
            logger.warning(f"Firestore history check failed: {e}. Proceeding without check.")
            return False

    def get_strategy(self):
        """
        Decides on the content strategy: Thread (Text) or Video.
        Returns a dict with 'type', 'content', 'topic', and optional 'video_prompt'.
        """
        topic = self._get_trending_topic()
        
        # Retry logic for duplicates
        if self._check_history(topic):
            logger.info("Duplicate topic detected. Requesting alternative.")
            prompt = f"The topic '{topic}' was already covered. Give me a DIFFERENT trending tech topic. Return ONLY the topic name."
            try:
                response = self.model.generate_content(prompt)
                new_topic = response.text.strip()
                # Check history again for the new topic
                if self._check_history(new_topic):
                     topic = "Python Coding Tips" # Ultimate fallback
                else:
                     topic = new_topic
            except Exception:
                topic = "Python Coding Tips" # Fallback
        
        logger.info(f"Selected Topic: {topic}")

        # Decide format
        if Config.BUDGET_MODE:
            post_type = "thread"
        else:
            # Ask Gemini if this topic is better for video or text
            decision_prompt = f"For the tech topic '{topic}', is it better to make a short video or a text thread? Reply with 'VIDEO' or 'THREAD'."
            try:
                decision = self.model.generate_content(decision_prompt).text.strip().upper()
                # Strict check
                if "VIDEO" in decision and "THREAD" not in decision:
                     post_type = "video"
                elif "VIDEO" in decision:
                     post_type = "video" # Lean towards video if mentioned
                else:
                     post_type = "thread"
            except Exception:
                post_type = "thread" # Default to thread on error

        strategy = {
            "topic": topic,
            "type": post_type,
            "timestamp": firestore.SERVER_TIMESTAMP # Use server timestamp
        }

        if post_type == "video":
            # Generate Video Prompt and Tweet Text
            script_prompt = f"Write a tweet caption for a video about '{topic}'. Also provide a visual prompt for an AI video generator. Format: CAPTION: <text> | PROMPT: <visual description>"
            try:
                response = self.model.generate_content(script_prompt).text.strip()
                if "|" in response:
                    parts = response.split("|")
                    caption = parts[0].replace("CAPTION:", "").strip()
                    visual_prompt = parts[1].replace("PROMPT:", "").strip()
                else:
                    # Fallback parsing
                    caption = response[:100]
                    visual_prompt = f"Tech visualization of {topic}"
            except Exception as e:
                logger.error(f"Failed to generate video script: {e}")
                # Fallback
                caption = f"Check out this update on {topic}! #tech #ai"
                visual_prompt = f"Futuristic technology visualization of {topic}, cinematic lighting, 4k"
            
            strategy["content"] = caption
            strategy["video_prompt"] = visual_prompt
            
        else:
            # Generate Thread
            thread_prompt = f"Write a 3-tweet thread about '{topic}' for a tech audience. Separate tweets with '|||'."
            try:
                response = self.model.generate_content(thread_prompt).text.strip()
                tweets = response.split("|||")
                cleaned_tweets = [t.strip() for t in tweets if t.strip()]
                if not cleaned_tweets:
                    cleaned_tweets = [f"Exciting news about {topic}! #tech"]
                strategy["content"] = cleaned_tweets
            except Exception as e:
                logger.error(f"Failed to generate thread: {e}")
                strategy["content"] = [f"Exciting news about {topic}! Stay tuned for more updates. #tech"]

        return strategy

    def log_post(self, strategy: dict, success: bool, error: str = None):
        """Logs the attempt to Firestore."""
        try:
            doc_ref = self.collection.document()
            data = strategy.copy()
            data["success"] = success
            if error:
                data["error"] = error
            # Ensure timestamp is set if it was missing (e.g. if strategy was created manually)
            if "timestamp" not in data:
                data["timestamp"] = firestore.SERVER_TIMESTAMP
            doc_ref.set(data)
        except Exception as e:
            logger.error(f"Failed to log post to Firestore: {e}")
