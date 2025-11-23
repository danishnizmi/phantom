import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import firestore
from config import Config
import datetime

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
        In a real scenario, this could be augmented with Google Search tool or scraping.
        For now, we rely on Gemini's internal knowledge or simulated "current" trends.
        """
        prompt = "Identify a single, specific, currently trending technology topic or news item suitable for a tech influencer tweet. Return ONLY the topic name."
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def _check_history(self, topic: str) -> bool:
        """
        Checks Firestore to see if we've recently posted about this topic.
        Returns True if we should SKIP this topic (duplicate), False otherwise.
        """
        # Simple check: query for recent posts with this topic in metadata
        # Note: This requires storing 'topic' in the document.
        
        # Let's look at the last 5 posts to see if the topic is mentioned
        docs = self.collection.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
        
        for doc in docs:
            data = doc.to_dict()
            if data.get("topic", "").lower() == topic.lower():
                print(f"Topic '{topic}' was recently covered. Skipping.")
                return True
        return False

    def get_strategy(self):
        """
        Decides on the content strategy: Thread (Text) or Video.
        Returns a dict with 'type', 'content', 'topic', and optional 'video_prompt'.
        """
        topic = self._get_trending_topic()
        
        if self._check_history(topic):
            # If duplicate, try one more time or just pick a generic evergreen topic
            topic = "Python Tips" # Fallback
        
        print(f"Selected Topic: {topic}")

        # Decide format
        # If BUDGET_MODE is True, always Thread.
        # Otherwise, flip a coin or ask Gemini what's best.
        if Config.BUDGET_MODE:
            post_type = "thread"
        else:
            # Ask Gemini if this topic is better for video or text
            decision_prompt = f"For the tech topic '{topic}', is it better to make a short video or a text thread? Reply with 'VIDEO' or 'THREAD'."
            decision = self.model.generate_content(decision_prompt).text.strip().upper()
            post_type = "video" if "VIDEO" in decision else "thread"

        strategy = {
            "topic": topic,
            "type": post_type,
            "timestamp": datetime.datetime.utcnow()
        }

        if post_type == "video":
            # Generate Video Prompt and Tweet Text
            script_prompt = f"Write a tweet caption for a video about '{topic}'. Also provide a visual prompt for an AI video generator. Format: CAPTION: <text> | PROMPT: <visual description>"
            response = self.model.generate_content(script_prompt).text.strip()
            
            # Parse response (naive parsing)
            try:
                parts = response.split("|")
                caption = parts[0].replace("CAPTION:", "").strip()
                visual_prompt = parts[1].replace("PROMPT:", "").strip()
            except:
                # Fallback
                caption = f"Check out this update on {topic}! #tech #ai"
                visual_prompt = f"Futuristic technology visualization of {topic}, cinematic lighting, 4k"
            
            strategy["content"] = caption
            strategy["video_prompt"] = visual_prompt
            
        else:
            # Generate Thread
            thread_prompt = f"Write a 3-tweet thread about '{topic}' for a tech audience. Separate tweets with '|||'."
            response = self.model.generate_content(thread_prompt).text.strip()
            tweets = response.split("|||")
            strategy["content"] = [t.strip() for t in tweets if t.strip()]

        return strategy

    def log_post(self, strategy: dict, success: bool, error: str = None):
        """Logs the attempt to Firestore."""
        doc_ref = self.collection.document()
        data = strategy.copy()
        data["success"] = success
        if error:
            data["error"] = error
        doc_ref.set(data)
