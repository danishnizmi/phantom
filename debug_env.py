import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Sys path: {sys.path}")

try:
    import vertexai
    print(f"Vertex AI version: {vertexai.__version__}")
    print(f"Vertex AI file: {vertexai.__file__}")
except ImportError as e:
    print(f"Failed to import vertexai: {e}")

try:
    from vertexai.preview.vision_models import ImageGenerationModel
    print("Successfully imported ImageGenerationModel")
except ImportError as e:
    print(f"Failed to import ImageGenerationModel: {e}")
