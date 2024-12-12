from dataclasses import dataclass

@dataclass
class RequestData:
    id: str
    price: float
    floor_price: float
    data_center: str
    app_publisher_id: str
    bundle_id: str
    tag_id: str
    device_geo_country: str
    ext_ad_format: str

@dataclass
class ResponseData:
    optimized_price: float
    status: str

@dataclass
class FeedBackRequest:
    id: str
    impression: bool
    price: float

@dataclass
class FeedBackResponse:
    ack: bool