from fastapi import APIRouter
from models import OptimizeRequest, OptimizeResponse, FeedbackRequest, FeedbackResponse
from state import SPACES, CACHE, context_hash
import time

router = APIRouter()

@router.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest):
    if request.floor_price <= 0 or request.price <=0 or request.floor_price > request.price:
        return OptimizeResponse(optimized_price=request.price, status="validation error")

    ctx = context_hash(request.data_center, request.bundle_id, request.tag_id, request.device_geo_country, request.ext_ad_format)
    space = SPACES.get(ctx)
    if not space:
        return OptimizeResponse(optimized_price=request.price, status="no space")

    # Пробуем чаще вызывать exploration
    exploration_price, ok = space.explore(request.floor_price, request.price)
    if ok:
        # explored
        buckets = space.sampleBuckets(exploration_price)
        edata = space.ExploreDataClass(ctx, buckets, time.time())
        ttl = space.short_ttl()
        CACHE[request.id] = edata
        return OptimizeResponse(optimized_price=exploration_price, status="explored")
    else:
        recommended_price = space.exploit(request.floor_price, request.price)
        return OptimizeResponse(optimized_price=recommended_price, status="exploited")

@router.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    val = CACHE.pop(request.id, None)
    if not val:
        return FeedbackResponse(ack=False)
    ctx = val.ContextHash
    space = SPACES.get(ctx)
    if not space:
        return FeedbackResponse(ack=False)
    space.Update(val, request.impression)
    return FeedbackResponse(ack=True)

@router.get("/space")
def space_handler(ctx: str):
    space = SPACES.get(ctx)
    if not space:
        return {"level":[]}
    return space.WC().dict()
