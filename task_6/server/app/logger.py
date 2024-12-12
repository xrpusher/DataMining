import logging
import sys
import os

def get_logger(level_str: str):
    level = logging.DEBUG
    if level_str == "info":
        level = logging.INFO
    elif level_str == "error":
        level = logging.ERROR
    elif level_str == "warning":
        level = logging.WARNING
    elif level_str == "debug":
        level = logging.DEBUG

    logger = logging.getLogger("app_logger")
    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] | %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False

    # Добавляем файл-логгер
    file_handler = logging.FileHandler("logs.txt", mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
