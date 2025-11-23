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
        
        # Multi-model configuration with dynamic discovery
        # Query Vertex AI to find available Gemini models instead of hardcoding
        candidate_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite", 
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite-001",
            "gemini-2.5-pro",
        ]
        
        self.model_names = []
        self.models = {}
        
        logger.info("Discovering available Gemini models...")
        
        # Test each candidate model to see if it's available
        for model_name in candidate_models:
            try:
                model = GenerativeModel(model_name)
                # Quick test to verify model is accessible
                test_response = model.generate_content("Hi", 
                    generation_config={"max_output_tokens": 5, "temperature": 0})
                
                if test_response.text:
                    self.models[model_name] = model
                    self.model_names.append(model_name)
                    logger.info(f"✓ Verified model: {model_name}")
                else:
                    logger.warning(f"⚠ {model_name} responded but empty")
                    
            except Exception as e:
                error_str = str(e)
                if "404" in error_str:
                    logger.debug(f"✗ {model_name} not available (404)")
                else:
                    logger.warning(f"✗ {model_name} failed: {error_str[:80]}")
        
        if not self.models:
            raise RuntimeError(f"No Gemini models available. Tried: {candidate_models}")
        
        logger.info(f"✓ Active models ({len(self.models)}): {self.model_names}")
        
        self.db = firestore.Client(project=self.project_id)
        self.collection = self.db.collection(Config.COLLECTION_NAME)

    def _generate_with_fallback(self, prompt: str) -> str:
        """
        Attempts to generate content using available models with fallback.
        Tries each model in order until successful or all fail.
        """
        last_error = None
        
        for model_name in self.model_names:
            if model_name not in self.models:
                continue
                
            try:
                model = self.models[model_name]
                response = model.generate_content(prompt)
                
                if response.text:
                    logger.info(f"✓ Generated content with {model_name}")
                    return response.text.strip()
                else:
                    logger.warning(f"✗ {model_name} returned empty response")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"✗ {model_name} failed: {str(e)[:100]}")
                continue
        
        # All models failed
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def _get_trending_topic(self) -> str:
        """
        Asks Gemini to identify a trending tech topic.
        """
        prompt = """Identify a single trending topic in tech right now. Focus on:
        - New AI tools or models (ChatGPT updates, Gemini, Claude, open-source LLMs)
        - Popular GitHub repositories or developer tools
        - Tech news (product launches, acquisitions, breakthroughs)
        Return ONLY the topic name, be specific (e.g., 'Cursor AI Editor' not 'AI Tools')."""
        return self._generate_with_fallback(prompt)

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
            prompt = f"""The topic '{topic}' was already covered. Give me a DIFFERENT trending tech topic. Focus on:
                - New AI tools/models, GitHub repos, or developer tools
                - Recent tech news or product launches
                Return ONLY the topic name."""
            try:
                new_topic = self._generate_with_fallback(prompt)
                # Check history again for the new topic
                if self._check_history(new_topic):
                     raise ValueError(f"Both '{topic}' and '{new_topic}' are duplicates. No fresh content available.")
                else:
                     topic = new_topic
            except Exception as e:
                logger.error(f"Failed to find alternative topic: {e}")
                raise # Fail instead of using fallback
        
        logger.info(f"Selected Topic: {topic}")

        # Decide format
        if Config.BUDGET_MODE:
            post_type = "thread"
        else:
            # Ask Gemini if this topic is better for video or text
            decision_prompt = f"""For the tech topic '{topic}', should we create a VIDEO or TEXT THREAD?
            Use VIDEO if:
            - The topic benefits from visual demonstration (UI walkthrough, code editor features, terminal commands)
            - It's a hook/teaser for something visual (app demo, GitHub repo showcase)
            - It needs to show "before/after" or comparisons
            
            Use THREAD if:
            - It's news, announcements, or opinion-based
            - It's a list, tips, or step-by-step guide better suited for text
            
            Reply ONLY with 'VIDEO' or 'THREAD'."""
            try:
                decision = self._generate_with_fallback(decision_prompt).upper()
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
                response = self._generate_with_fallback(script_prompt)
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
                raise # Fail instead of using fallback
            
            strategy["content"] = caption
            strategy["video_prompt"] = visual_prompt
            
        else:
            # Generate Thread
            thread_prompt = f"""Write a 3-tweet thread about '{topic}' for a tech influencer audience.
            Focus on:
            - Why developers should care
            - Key features or benefits
            - Call-to-action (try it, check it out, share thoughts)
            
            Separate tweets with '|||'. Keep each tweet under 260 characters."""
            try:
                response = self._generate_with_fallback(thread_prompt)
                tweets = response.split("|||")
                cleaned_tweets = [t.strip() for t in tweets if t.strip()]
                if not cleaned_tweets:
                    # Create unique fallback using timestamp and randomization to avoid duplicate content errors
                    import random
                    emojis = ["🚀", "💡", "🔥", "✨", "⚡"]
                    timestamp_str = datetime.datetime.now().strftime("%H%M")
                    cleaned_tweets = [f"{random.choice(emojis)} Diving into {topic} today! Stay tuned... #{timestamp_str} #tech"]
                strategy["content"] = cleaned_tweets
            except Exception as e:
                logger.error(f"Failed to generate thread: {e}")
                raise # Fail instead of using fallback

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
