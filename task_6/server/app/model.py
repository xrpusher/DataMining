from dataclasses import dataclass

# Data class representing a request with various fields.
@dataclass
class RequestData:
    id: str                     # Unique identifier for the request
    price: float                # Price value in the request
    floor_price: float          # Minimum price value
    data_center: str            # Data center associated with the request
    app_publisher_id: str       # ID of the application publisher
    bundle_id: str              # ID of the application bundle
    tag_id: str                 # Tag identifier for categorization
    device_geo_country: str     # Country information for the device
    ext_ad_format: str          # External advertisement format

# Data class for the response, containing the optimized price and status.
@dataclass
class ResponseData:
    optimized_price: float      # Optimized price calculated by the server
    status: str                 # Status message indicating the result

# Data class for feedback requests sent to update exploration results.
@dataclass
class FeedBackRequest:
    id: str                     # ID of the request being referenced
    impression: bool            # Whether the ad was successful (impression)
    price: float                # Price at which the ad was shown

# Data class for feedback responses indicating acknowledgment.
@dataclass
class FeedBackResponse:
    ack: bool                   # Acknowledgment flag for the feedback