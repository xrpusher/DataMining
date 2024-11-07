from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random

app = FastAPI()

class BidRequest(BaseModel):
    slot_id: str
    bid_price: float
    base_price: float

@app.post("/evaluate_bid/")
async def evaluate_bid(bid_request: BidRequest):
    """
    Оценивает ставку на основе прайса и возвращает win или loss.
    """
    # Порог выигрыша, зависящий от базовой цены и случайного множителя
    win_threshold = bid_request.base_price * random.uniform(1.1, 1.3)
    
    # Решаем, выиграна ли ставка
    if bid_request.bid_price >= win_threshold:
        return {"result": "win"}
    else:
        return {"result": "loss"}
