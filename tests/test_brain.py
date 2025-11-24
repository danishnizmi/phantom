import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock environment variables before importing config
os.environ["PROJECT_ID"] = "test-project"
os.environ["REGION"] = "us-central1"

# Mock imports that require GCP credentials
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.firestore"] = MagicMock()
sys.modules["vertexai"] = MagicMock()
sys.modules["vertexai.generative_models"] = MagicMock()
sys.modules["vertexai.preview"] = MagicMock()
sys.modules["vertexai.preview.vision_models"] = MagicMock()

from brain import AgentBrain

class TestAgentBrain(unittest.TestCase):
    
    @patch("brain.firestore.Client")
    @patch("brain.GenerativeModel")
    def setUp(self, mock_model, mock_firestore):
        self.mock_model_instance = mock_model.return_value
        self.mock_db_instance = mock_firestore.return_value
        self.brain = AgentBrain()

    def test_get_trending_topic_success(self):
        # Setup mock
        mock_response = MagicMock()
        mock_response.text = "AI Agents"
        self.mock_model_instance.generate_content.return_value = mock_response
        
        # Test
        topic = self.brain._get_trending_topic()
        self.assertEqual(topic, "AI Agents")

    def test_get_trending_topic_failure(self):
        # Setup mock to raise exception
        self.mock_model_instance.generate_content.side_effect = Exception("API Error")
        
        # Test fallback
        # Test fallback - now expects exception as _generate_with_fallback raises RuntimeError when all fail
        with self.assertRaises(RuntimeError):
            self.brain._get_trending_topic()

    def test_check_history_duplicate(self):
        # Setup mock Firestore
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {"topic": "AI Agents"}
        
        mock_stream = MagicMock()
        mock_stream.stream.return_value = [mock_doc]
        self.brain.collection.order_by.return_value.limit.return_value = mock_stream
        
        # Test
        is_duplicate = self.brain._check_history("AI Agents")
        self.assertTrue(is_duplicate)

    def test_check_history_new(self):
        # Setup mock Firestore
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {"topic": "Python"}
        
        mock_stream = MagicMock()
        mock_stream.stream.return_value = [mock_doc]
        self.brain.collection.order_by.return_value.limit.return_value = mock_stream
        
        # Test
        is_duplicate = self.brain._check_history("AI Agents")
        self.assertFalse(is_duplicate)

if __name__ == "__main__":
    unittest.main()
