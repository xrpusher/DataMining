import os
from flask import Flask
from app.config import load_config
from app.logger import get_logger
from app.handlers import register_handlers
from app.space import load_spaces
from app.misc import Cache
import atexit
import threading
import time

Spaces = {}
Cache_ = None
TLog = None

def create_app():
    global Spaces, Cache_, TLog
    cfg = load_config()
    TLog = get_logger(cfg.log_level)

    TLog.info(f"Loaded config: {cfg}")
    Cache_ = Cache(ttl_seconds=cfg.cache_ttl, logger=TLog)
    Spaces = load_spaces(cfg, TLog)

    # Не вызываем s.start_bg_task(), не запускаем никакие глобальные background_tasks
    TLog.debug("Initialization done")

    app = Flask(__name__)
    register_handlers(app, Spaces, Cache_, TLog)

    return app

app = create_app()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
