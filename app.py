import taipy as tp
from taipy import Gui
import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Taipy pages
root_md = """
# Hello Taipy

This is a test page.

<|button|label=Click me|on_action=on_button_click|>

Current count: <|{count}|>
"""

# Function to handle button click
def on_button_click(state):
    logger.debug("Button clicked")
    state.count += 1

# Initial state
initial_state = {
    "count": 0
}

# Create the Gui object
gui = Gui(pages={"/": root_md})  # Changed from "" to "/"

if __name__ == "__main__":
    logger.debug("Starting Taipy app")
    port = int(os.environ.get("PORT", 8080))
    logger.debug(f"Using port: {port}")
    
    try:
        gui.run(host="0.0.0.0", port=port, use_reloader=False, debug=True)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(f"Sys.path: {sys.path}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Directory contents: {os.listdir()}")
        raise
