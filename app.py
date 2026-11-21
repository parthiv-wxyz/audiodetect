import os
import sys

# Ensure root directory is on the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from gradio_app import demo, theme, custom_css

if __name__ == "__main__":
    # Hugging Face Spaces and other cloud platforms handle port 7860 and host 0.0.0.0 automatically
    demo.launch(
        theme=theme,
        css=custom_css,
        show_error=True
    )
