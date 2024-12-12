import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    buffer_size: int
    discount: float
    desired_exploration_speed: float
    log_level: str
    level_size: int
    bucket_size: int
    space_desc_file: str
    cache_ttl: int

def load_config():
    # Создаём базовый объект конфигурации
    config = Config(
        buffer_size=int(os.getenv("BUFFER_SIZE", "100")),
        discount=float(os.getenv("DISCOUNT", "0.25")),
        desired_exploration_speed=float(os.getenv("DESIRED_EXPLORATION_SPEED", "2")),
        log_level=os.getenv("LOG_LEVEL", "trace"),
        level_size=int(os.getenv("LEVEL_SIZE", "5")),
        bucket_size=int(os.getenv("BUCKET_SIZE", "30")),
        space_desc_file=os.getenv("SPACE_DESC_FILE", ""),  # Оставляем пустым временно
        cache_ttl=int(os.getenv("CACHE_TTL", "10"))
    )

    # Если путь к SPACE_DESC_FILE не задан в .env, указываем относительный путь
    if not config.space_desc_file:
        # Получаем путь к папке server (уровень выше папки app)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config.space_desc_file = os.path.join(base_dir, "data", "spaces_desc.json")

    return config
