import requests
import time
import random

API_URL = "http://localhost:8000/evaluate_bid/"

def calculate_bid(base_price, adjustment_factor):
    """
    Рассчитывает ставку на основе базовой цены и корректирующего коэффициента.
    """
    return round(base_price * adjustment_factor, 2)

def dsp_advert_experiment():
    win_count = 0
    loss_count = 0
    adjustment_factor = 1.1  # начальный коэффициент, например, ставка на 10% выше базовой цены
    experiment_iterations = 20  # количество итераций для эксперимента

    for i in range(experiment_iterations):
        # Создаем данные для рекламного слота
        slot_details = {
            'slot_id': f'slot_{i + 1}',
            'base_price': random.uniform(50, 200)  # случайная базовая цена от 50 до 200 долларов
        }
        
        # Рассчитываем ставку
        bid_price = calculate_bid(slot_details['base_price'], adjustment_factor)
        
        # Подготовка данных для отправки на сервер
        bid_request = {
            "slot_id": slot_details['slot_id'],
            "bid_price": bid_price,
            "base_price": slot_details['base_price']
        }

        # Отправка запроса на преподавательский сервер
        response = requests.post(API_URL, json=bid_request)
        
        if response.status_code == 200:
            result = response.json().get("result")
            
            # Обработка результата
            if result == "win":
                win_count += 1
                print(f"Round {i + 1}: WIN with bid ${bid_price} for slot {slot_details['slot_id']} (base ${slot_details['base_price']})")
                # Снижаем коэффициент для экономии бюджета
                adjustment_factor = max(1.05, adjustment_factor - 0.02)
            else:
                loss_count += 1
                print(f"Round {i + 1}: LOSS with bid ${bid_price} for slot {slot_details['slot_id']} (base ${slot_details['base_price']})")
                # Увеличиваем коэффициент, чтобы повысить вероятность выигрыша
                adjustment_factor += 0.05
        else:
            print(f"Error: Received status code {response.status_code}")

        time.sleep(0.5)  # пауза для имитации времени обработки

    # Вывод результатов эксперимента
    print(f"\nTotal Wins: {win_count}, Total Losses: {loss_count}")
    print(f"Final Adjustment Factor: {adjustment_factor}")

# Запуск экспериментального цикла
if __name__ == "__main__":
    dsp_advert_experiment()
