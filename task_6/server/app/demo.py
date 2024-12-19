import requests
import uuid
import random
import time

# Пример имитации запросов к /optimize и /feedback
# Предположим, что сервер запущен локально на 127.0.0.1:8000

def run_demo(num_requests=100):
    # Допустим известный контекст
    ctx = "13677617117914323147"  
    # Генерируем запросы
    base_url = "http://127.0.0.1:8000"
    floor_price = 1.0
    max_price = 10.0

    for i in range(num_requests):
        rid = str(uuid.uuid4())
        price = random.uniform(floor_price, max_price)
        body = {
            "id": rid,
            "price": price,
            "floor_price": floor_price,
            "data_center": "dc1",
            "app_publisher_id": "pub1",
            "bundle_id": "bundle1",
            "tag_id": "tag1",
            "device_geo_country": "US",
            "ext_ad_format": "video"
        }
        resp = requests.post(f"{base_url}/optimize", json=body).json()
        new_price = resp["optimized_price"]
        # Имитируем вероятности импрессии (выигрыша)
        impression = random.random() < 0.5  # например 50%
        fb_body = {
            "id": rid,
            "impression": impression,
            "price": new_price
        }
        requests.post(f"{base_url}/feedback", json=fb_body)

    # Запросим статистику
    eval_resp = requests.get(f"{base_url}/evaluate?ctx={ctx}").json()
    print("Evaluation:", eval_resp)

if __name__ == "__main__":
    run_demo()
