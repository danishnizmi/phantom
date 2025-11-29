import logging
import tempfile
from typing import Dict, List, Optional
from vertexai.preview.vision_models import ImageGenerationModel

logger = logging.getLogger(__name__)

class InfographicGenerator:
    """
    Generates educational infographics using Vertex AI Imagen.
    Creates visually appealing explanatory graphics for tech topics.
    """

    # Imagen models to try (in order of preference)
    # See: https://cloud.google.com/vertex-ai/generative-ai/docs/image/generate-images
    IMAGEN_MODELS = [
        "imagen-4.0-fast-generate-001",  # Imagen 4 Fast - newest, best latency
        "imagen-4.0-generate-001",       # Imagen 4 - high quality
        "imagen-3.0-fast-generate-001",  # Imagen 3 Fast - fallback
        "imagen-3.0-generate-001",       # Imagen 3 - fallback
    ]

    def __init__(self):
        self.model = None
        self.model_name = None
        self._init_model()

        # Infographic style templates
        self.styles = {
            'flowchart': {
                'description': 'Clean flowchart diagram with connected boxes and arrows',
                'colors': 'professional blue and white color scheme',
                'elements': 'geometric shapes, connecting lines, numbered steps'
            },
            'comparison': {
                'description': 'Side-by-side comparison layout',
                'colors': 'contrasting colors for two sides',
                'elements': 'split layout, icons, bullet points, versus symbol'
            },
            'timeline': {
                'description': 'Horizontal or vertical timeline',
                'colors': 'gradient progression from past to future',
                'elements': 'dated markers, milestone icons, connecting line'
            },
            'data_viz': {
                'description': 'Data visualization with charts and numbers',
                'colors': 'vibrant data colors on dark background',
                'elements': 'bar charts, pie charts, percentages, trend lines'
            },
            'explainer': {
                'description': 'Central concept with radiating explanations',
                'colors': 'warm center, cool periphery',
                'elements': 'central icon, branching connections, label boxes'
            },
            'process': {
                'description': 'Step-by-step process diagram',
                'colors': 'sequential color progression',
                'elements': 'numbered circles, directional arrows, action verbs'
            },
            'layered': {
                'description': 'Layered architecture or stack diagram',
                'colors': 'distinct layer colors, gradient depth',
                'elements': 'stacked rectangles, layer labels, connection indicators'
            }
        }

    def _init_model(self):
        """Initialize Imagen model with fallback to older versions."""
        for model_name in self.IMAGEN_MODELS:
            try:
                self.model = ImageGenerationModel.from_pretrained(model_name)
                self.model_name = model_name
                logger.info(f"✓ Imagen model initialized: {model_name}")
                return
            except Exception as e:
                logger.warning(f"✗ Failed to initialize {model_name}: {e}")
                continue

        raise RuntimeError(f"Failed to initialize any Imagen model. Tried: {self.IMAGEN_MODELS}")

    def select_style(self, topic: str, context: str = "") -> str:
        """
        Selects the best infographic style based on topic.
        """
        topic_lower = topic.lower()
        context_lower = context.lower()
        combined = topic_lower + " " + context_lower

        # Pattern matching for style selection
        if any(kw in combined for kw in ['vs', 'versus', 'compare', 'difference', 'better']):
            return 'comparison'
        elif any(kw in combined for kw in ['history', 'evolution', 'timeline', 'year', 'decade']):
            return 'timeline'
        elif any(kw in combined for kw in ['step', 'process', 'how to', 'guide', 'workflow']):
            return 'process'
        elif any(kw in combined for kw in ['data', 'statistics', 'numbers', 'percent', 'growth', 'market']):
            return 'data_viz'
        elif any(kw in combined for kw in ['architecture', 'stack', 'layer', 'component']):
            return 'layered'
        elif any(kw in combined for kw in ['flow', 'pipeline', 'system']):
            return 'flowchart'
        else:
            return 'explainer'

    def generate_infographic_prompt(
        self,
        topic: str,
        key_points: List[str] = None,
        style: str = None,
        context: str = ""
    ) -> str:
        """
        Generates a detailed prompt for Imagen to create an infographic.
        """
        if style is None:
            style = self.select_style(topic, context)

        style_info = self.styles.get(style, self.styles['explainer'])

        # Build key points section
        points_text = ""
        if key_points:
            points_text = f"Key concepts to visualize: {', '.join(key_points[:5])}"

        prompt = f"""Professional tech infographic about "{topic}".

Style: {style_info['description']}
Color scheme: {style_info['colors']}
Visual elements: {style_info['elements']}

{points_text}

Requirements:
- Clean, modern design suitable for social media
- High contrast, readable at small sizes
- Minimalist tech aesthetic
- No text clutter, let visuals explain
- Professional quality suitable for tech publication
- 16:9 aspect ratio optimized for Twitter/X

Do NOT include:
- Watermarks
- Stock photo elements
- Generic clip art
- Busy or cluttered layouts"""

        logger.info(f"Generated infographic prompt for style '{style}': {prompt[:100]}...")
        return prompt

    def generate(
        self,
        topic: str,
        key_points: List[str] = None,
        style: str = None,
        context: str = ""
    ) -> str:
        """
        Generates an infographic image and saves to temp file.
        Returns path to the saved image.
        """
        prompt = self.generate_infographic_prompt(topic, key_points, style, context)

        try:
            logger.info(f"Generating infographic for: {topic[:50]}...")

            images = self.model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio="16:9",
                safety_filter_level="block_medium_and_above",  # Per docs: block_some is deprecated
                person_generation="dont_allow"  # Infographics shouldn't need people
            )

            if not images:
                raise ValueError("No infographic generated")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                output_path = tmp_file.name

            images[0].save(location=output_path, include_generation_parameters=False)
            logger.info(f"Infographic saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Infographic generation failed: {e}")
            raise

    def generate_caption(
        self,
        topic: str,
        key_points: List[str] = None,
        source_url: str = None
    ) -> str:
        """
        Generates a caption for the infographic post.
        Keeps it concise and informative.
        """
        # Base caption with topic
        base = topic if len(topic) <= 100 else topic[:97] + "..."

        # Add key points hint if available
        if key_points and len(key_points) >= 2:
            hint = f"Breaking down: {key_points[0]} and {key_points[1]}"
            if len(base) + len(hint) + 4 <= 175:
                base = f"{hint}.\n\n{base}"

        # Add URL if provided
        if source_url:
            if len(base) + len(source_url) + 4 <= 280:
                base = f"{base}\n\n{source_url}"
            else:
                max_len = 280 - len(source_url) - 7
                base = f"{base[:max_len]}...\n\n{source_url}"

        return base


class InfographicStrategy:
    """
    High-level strategy for creating infographic posts.
    Integrates with YouTube fetcher for topic discovery.
    """

    def __init__(self, generator: InfographicGenerator = None, youtube_fetcher=None):
        self.generator = generator or InfographicGenerator()
        self.youtube_fetcher = youtube_fetcher

    def create_infographic_post(
        self,
        topic: str = None,
        key_points: List[str] = None,
        source_url: str = None,
        context: str = ""
    ) -> Dict:
        """
        Creates a complete infographic post with image and caption.
        Returns dict with 'image_path', 'caption', 'topic', 'type'.
        """
        # If no topic provided, try to get one from YouTube
        if not topic and self.youtube_fetcher:
            video = self.youtube_fetcher.get_infographic_topic()
            if video:
                topic = video['title']
                key_points = self.youtube_fetcher.extract_key_concepts(video)
                source_url = video.get('url')
                context = video.get('description', '')

        if not topic:
            raise ValueError("No topic provided and couldn't fetch from YouTube")

        # Select style and generate
        style = self.generator.select_style(topic, context)
        image_path = self.generator.generate(topic, key_points, style, context)
        caption = self.generator.generate_caption(topic, key_points, source_url)

        return {
            'type': 'infographic',
            'image_path': image_path,
            'content': caption,
            'topic': topic,
            'key_points': key_points,
            'style': style,
            'source_url': source_url,
            'image_prompt': self.generator.generate_infographic_prompt(topic, key_points, style, context)
        }
