import os
from flask import Flask
from app.config import load_config       # Load application configurations
from app.logger import get_logger       # Set up application logging
from app.handlers import register_handlers  # Register Flask routes/handlers
from app.space import load_spaces       # Load spaces for processing
from app.misc import Cache              # Import Cache class
import atexit
import threading
import time

# Global variables to hold shared application resources.
Spaces = {}    # Dictionary to store "spaces" for processing requests
Cache_ = None  # Global cache object
TLog = None    # Global logger object

# Function to create and initialize the Flask application.
def create_app():
    global Spaces, Cache_, TLog
    cfg = load_config()                 # Load configuration from .env or defaults
    TLog = get_logger(cfg.log_level)    # Set up logging with the specified log level

    TLog.info(f"Loaded config: {cfg}")
    Cache_ = Cache(ttl_seconds=cfg.cache_ttl, logger=TLog)  # Initialize the cache
    Spaces = load_spaces(cfg, TLog)     # Load spaces from the configuration

    # Log that initialization is complete.
    TLog.debug("Initialization done")

    # Create the Flask application instance.
    app = Flask(__name__)
    register_handlers(app, Spaces, Cache_, TLog)  # Register HTTP request handlers

    return app  # Return the initialized Flask app

# Create the application instance.
app = create_app()

# Entry point for running the application.
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)  # Run the Flask app on port 8000