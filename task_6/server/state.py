import hashlib
from space import Space, LoadSpaces
from misc import Config

CACHE = {}
SPACES = {}

def context_hash(dc: str, bundle: str, tag: str, geo: str, ad_format: str) -> str:
    h = hashlib.sha1((dc+bundle+tag+geo+ad_format).encode()).hexdigest()
    return h[:16]

def init_spaces(log):
    cfg = Config()
    cfg.BUFFER_SIZE = 10
    cfg.DISCOUNT = 0.0
    cfg.DESIRED_EXPLORATION_SPEED = 2.0
    cfg.LOG_LEVEL = "debug"
    cfg.LEVEL_SIZE = 3
    cfg.BUCKET_SIZE = 20
    cfg.SPACE_DESC_FILE = "spaces_desc.json"
    
    global SPACES
    SPACES, err = LoadSpaces(cfg, log)
    if err is not None:
        raise err
