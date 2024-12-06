from fastapi import FastAPI
from routes import router
from state import init_spaces
import logging

app = FastAPI()

log = logging.getLogger("server")
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
log.addHandler(ch)

init_spaces(log)  # Инициализируем Spaces из файла spaces_desc.json

app.include_router(router)

# Запуск:
# uvicorn main:app --reload
