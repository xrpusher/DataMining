from flask import request, jsonify
from app.model import RequestData, ResponseData, FeedBackRequest, FeedBackResponse
from app.misc import ValidationError
import json

# Function to register HTTP routes (handlers) for the Flask app
def register_handlers(app, Spaces, Cache_, TLog):

    # Endpoint for optimizing prices
    @app.route("/optimize", methods=["POST"])
    def optimize_handler():
        # Ensure the method is POST
        if request.method != "POST":
            return "Method not allowed", 405
        try:
            # Parse the request body as JSON and create a RequestData object
            body = request.get_json()
            r = RequestData(
                id=body["id"],
                price=body["price"],
                floor_price=body["floor_price"],
                data_center=body["data_center"],
                app_publisher_id=body["app_publisher_id"],
                bundle_id=body["bundle_id"],
                tag_id=body["tag_id"],
                device_geo_country=body["device_geo_country"],
                ext_ad_format=body["ext_ad_format"]
            )
            # Call the optimize function and return the response
            response = optimize(r, Spaces, Cache_, TLog)
            return jsonify({"optimized_price": response.optimized_price, "status": response.status}), 200
        except ValidationError:
            # Handle validation errors
            return jsonify({"optimized_price": 0.0, "status": "validation error"}), 200
        except Exception as e:
            # Log any unexpected errors and return a generic error response
            TLog.error(f"Error in optimize: {str(e)}")
            return jsonify({"optimized_price": 0.0, "status": str(e)}), 200

    # Endpoint for processing feedback
    @app.route("/feedback", methods=["POST"])
    def feedback_handler():
        if request.method != "POST":
            return "Method not allowed", 405
        # Parse the request body as JSON and create a FeedBackRequest object
        body = request.get_json()
        fb = FeedBackRequest(
            id=body["id"],
            impression=body["impression"],
            price=body["price"]
        )
        # Call the update function and return the acknowledgment
        response = update(fb, Spaces, Cache_, TLog)
        return jsonify({"ack": response.ack}), 200

    # Endpoint for retrieving space data
    @app.route("/space", methods=["GET"])
    def space_handler():
        # Get the context hash from the query parameters
        ctx = request.args.get("ctx", "")
        # Prepare and return the space data
        resp = prepareSpace(ctx, Spaces)
        return jsonify(resp), 200


# Function to optimize pricing based on input data
def optimize(r: RequestData, Spaces, Cache_, TLog):
    # Validate the request data
    if not validate(r):
        TLog.error(f"validation error: {r.id}")
        return ResponseData(optimized_price=r.price, status="validation error")
    # Attempt to explore a new price or exploit existing data
    exp_price, ok, err = explore(r, Spaces, Cache_, TLog)
    if err:
        return ResponseData(optimized_price=r.price, status=str(err))
    if not ok:
        # If exploration fails, fallback to exploitation
        rec_price, err = exploit(r, Spaces, TLog)
        if err:
            return ResponseData(optimized_price=r.price, status=str(err))
        return ResponseData(optimized_price=rec_price, status="exploited")
    return ResponseData(optimized_price=exp_price, status="explored")


# Function to explore potential prices
def explore(r: RequestData, Spaces, Cache_, TLog):
    ctx = context_hash(r)
    # Check if the space exists
    if ctx not in Spaces:
        from app.misc import NoSpaceError
        return 0.0, False, NoSpaceError()

    s = Spaces[ctx]
    # Call the exploration logic in the space object
    exp_price, data, ttl, ok, err = s.explore(r.floor_price, r.price)
    if err:
        TLog.error(f"explore ctx: {ctx} error: {err}")
        return 0.0, False, err
    if not ok:
        TLog.debug(f"ctx: {ctx} NO exploration for price: {r.price}")
        return 0.0, False, None

    # Define a callback for cache expiration
    def cb(d):
        cctx = d["ContextHash"]
        ss = Spaces.get(cctx, None)
        if not ss:
            return False
        ss.update(d["Data"], False)
        return True

    # Cache the exploration result
    Cache_.set(r.id, {"ContextHash": data.ContextHash, "Data": data}, ttl, cb)
    TLog.debug(f"explore: {ctx} price: {r.price} new_price: {exp_price} ttl: {ttl}")
    return exp_price, True, None

# Function to exploit existing data for optimal pricing
def exploit(r: RequestData, Spaces, TLog):
    ctx = context_hash(r)
    s = Spaces.get(ctx, None)
    if not s:
        from app.misc import NoSpaceError
        return r.price, NoSpaceError()
    # Call the exploit logic in the space object
    rec, err = s.exploit(r.floor_price, r.price)
    if err:
        TLog.error(f"exploit ctx: {ctx} error: {err}")
        return r.price, err
    return rec, None

# Function to validate the input data
def validate(r: RequestData):
    if r.floor_price <= 0:
        return False
    if r.price <= 0:
        return False
    if r.floor_price > r.price:
        return False
    return True

# Implementation of the FNV-1a hash algorithm
import hashlib
def fnv64a(data, h=0xcbf29ce484222325):
    fnv_prime = 0x100000001b3
    for b in data:
        h ^= b
        h *= fnv_prime
        h &= 0xffffffffffffffff
    return h

# Generate a context hash for the request
def context_hash(r: RequestData):
    # Create a consistent hash string using relevant fields from the request
    s = (r.data_center + r.bundle_id + r.tag_id + r.device_geo_country + r.ext_ad_format + r.ext_ad_format).encode('utf-8')
    h = fnv64a(s)
    return str(h)

# Update the space data based on feedback
def update(fb: FeedBackRequest, Spaces, Cache_, TLog):
    val, ok = Cache_.pop(fb.id)
    if not ok:
        return FeedBackResponse(ack=False)
    cctx = val["ContextHash"]
    s = Spaces.get(cctx, None)
    if not s:
        return FeedBackResponse(ack=False)
    # Update the space with the feedback data
    s.update(val["Data"], fb.impression)
    return FeedBackResponse(ack=True)

# Prepare the space data for response
def prepareSpace(ctx, Spaces):
    s = Spaces.get(ctx, None)
    if not s:
        return {"level":[]}
    return s.wc()