from pydantic import BaseModel
from typing import List

class OptimizeRequest(BaseModel):
    id: str
    price: float
    floor_price: float
    data_center: str
    app_publisher_id: str
    bundle_id: str
    tag_id: str
    device_geo_country: str
    ext_ad_format: str

class OptimizeResponse(BaseModel):
    optimized_price: float
    status: str

class FeedbackRequest(BaseModel):
    id: str
    impression: bool
    price: float

class FeedbackResponse(BaseModel):
    ack: bool

class LevelEstimation(BaseModel):
    price: List[float]
    pr: List[float]

class LearnedEstimation(BaseModel):
    level: List[LevelEstimation]
