"""
Script to check available Gemini models in Vertex AI
"""
import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID = "phantom-479109"
REGION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)

# List of known Gemini models to test
models_to_test = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro",
    "gemini-1.5-pro-002",
    "gemini-2.0-flash-exp",
    "gemini-1.0-pro",  # deprecated, should fail
]

print("Testing Vertex AI Gemini Models:\n")
print(f"Project: {PROJECT_ID}")
print(f"Region: {REGION}\n")
print("-" * 80)

available_models = []

for model_name in models_to_test:
    try:
        model = GenerativeModel(model_name)
        # Try a simple test call
        response = model.generate_content("Hello", 
            generation_config={"max_output_tokens": 10, "temperature": 0})
        
        if response.text:
            available_models.append(model_name)
            print(f"✅ {model_name:30} - AVAILABLE")
        else:
            print(f"⚠️  {model_name:30} - No response")
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            print(f"❌ {model_name:30} - NOT FOUND")
        elif "403" in error_msg or "permission" in error_msg.lower():
            print(f"🔒 {model_name:30} - PERMISSION DENIED")
        else:
            print(f"⚠️  {model_name:30} - ERROR: {error_msg[:50]}")

print("\n" + "-" * 80)
print(f"\nAvailable models ({len(available_models)}):")
for model in available_models:
    print(f"  - {model}")
