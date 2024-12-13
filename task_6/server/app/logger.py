import logging
import sys
import os

# Function to configure and return a logger instance
def get_logger(level_str: str):
    # Map string log levels to logging module levels
    level = logging.DEBUG  # Default to DEBUG
    if level_str == "info":
        level = logging.INFO
    elif level_str == "error":
        level = logging.ERROR
    elif level_str == "warning":
        level = logging.WARNING
    elif level_str == "debug":
        level = logging.DEBUG

    # Create a logger instance
    logger = logging.getLogger("app_logger")
    logger.setLevel(level)  # Set the logging level

    # Configure the console handler (logs to stdout)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] | %(message)s')
    ch.setFormatter(formatter)  # Apply the formatting
    logger.addHandler(ch)  # Add the handler to the logger
    logger.propagate = False  # Prevent propagation to the root logger

    # Configure the file handler (logs to a file)
    file_handler = logging.FileHandler("logs.txt", mode='a')
    file_handler.setFormatter(formatter)  # Apply the same formatting
    logger.addHandler(file_handler)  # Add the file handler

    return logger  # Return the configured logger