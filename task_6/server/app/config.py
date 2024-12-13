import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Dataclass to store configuration parameters
@dataclass
class Config:
    buffer_size: int                  # Buffer size for caching
    discount: float                   # Discount factor for exploration/exploitation
    desired_exploration_speed: float  # Target exploration speed
    log_level: str                    # Logging level (e.g., debug, info)
    level_size: int                   # Number of levels for price buckets
    bucket_size: int                  # Number of buckets per level
    space_desc_file: str              # Path to the spaces description JSON file
    cache_ttl: int                    # Time-to-live for cached items

# Function to load the configuration
def load_config():
    # Create a Config instance with values from environment variables or defaults
    config = Config(
        buffer_size=int(os.getenv("BUFFER_SIZE", "100")),
        discount=float(os.getenv("DISCOUNT", "0.25")),
        desired_exploration_speed=float(os.getenv("DESIRED_EXPLORATION_SPEED", "2")),
        log_level=os.getenv("LOG_LEVEL", "trace"),
        level_size=int(os.getenv("LEVEL_SIZE", "5")),
        bucket_size=int(os.getenv("BUCKET_SIZE", "30")),
        space_desc_file=os.getenv("SPACE_DESC_FILE", ""),  # Initially empty
        cache_ttl=int(os.getenv("CACHE_TTL", "10"))
    )

    # Set the default path for the space description file if not provided
    if not config.space_desc_file:
        # Determine the base directory (parent of the `app` directory)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Define the relative path to the spaces_desc.json file
        config.space_desc_file = os.path.join(base_dir, "data", "spaces_desc.json")

    return config  # Return the loaded configuration