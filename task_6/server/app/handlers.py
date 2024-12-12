from flask import request, jsonify
from app.model import RequestData, ResponseData, FeedBackRequest, FeedBackResponse
from app.misc import ValidationError
import json

def register_handlers(app, Spaces, Cache_, TLog):

    @app.route("/optimize", methods=["POST"])
    def optimize_handler():
        if request.method != "POST":
            return "Method not allowed", 405
        try:
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
            response = optimize(r, Spaces, Cache_, TLog)
            return jsonify({"optimized_price": response.optimized_price, "status": response.status}), 200
        except ValidationError:
            return jsonify({"optimized_price": 0.0, "status": "validation error"}), 200
        except Exception as e:
            TLog.error(f"Error in optimize: {str(e)}")
            return jsonify({"optimized_price": 0.0, "status": str(e)}), 200

    @app.route("/feedback", methods=["POST"])
    def feedback_handler():
        if request.method != "POST":
            return "Method not allowed", 405
        body = request.get_json()
        fb = FeedBackRequest(
            id=body["id"],
            impression=body["impression"],
            price=body["price"]
        )
        response = update(fb, Spaces, Cache_, TLog)
        return jsonify({"ack": response.ack}), 200

    @app.route("/space", methods=["GET"])
    def space_handler():
        ctx = request.args.get("ctx", "")
        resp = prepareSpace(ctx, Spaces)
        return jsonify(resp), 200


def optimize(r: RequestData, Spaces, Cache_, TLog):
    if not validate(r):
        TLog.error(f"validation error: {r.id}")
        return ResponseData(optimized_price=r.price, status="validation error")
    # explore or exploit
    exp_price, ok, err = explore(r, Spaces, Cache_, TLog)
    if err:
        return ResponseData(optimized_price=r.price, status=str(err))
    if not ok:
        # exploit
        rec_price, err = exploit(r, Spaces, TLog)
        if err:
            return ResponseData(optimized_price=r.price, status=str(err))
        return ResponseData(optimized_price=rec_price, status="exploited")
    return ResponseData(optimized_price=exp_price, status="explored")


def explore(r: RequestData, Spaces, Cache_, TLog):
    ctx = context_hash(r)
    if ctx not in Spaces:
        from app.misc import NoSpaceError
        return 0.0, False, NoSpaceError()

    s = Spaces[ctx]
    exp_price, data, ttl, ok, err = s.explore(r.floor_price, r.price)
    if err:
        TLog.error(f"explore ctx: {ctx} error: {err}")
        return 0.0, False, err
    if not ok:
        TLog.debug(f"ctx: {ctx} NO exploration for price: {r.price}")
        return 0.0, False, None
    def cb(d):
        # callback if expires
        cctx = d["ContextHash"]
        ss = Spaces.get(cctx, None)
        if not ss:
            return False
        ss.update(d["Data"], False)
        return True
    Cache_.set(r.id, {"ContextHash": data.ContextHash, "Data": data}, ttl, cb)
    TLog.debug(f"explore: {ctx} price: {r.price} new_price: {exp_price} ttl: {ttl}")
    return exp_price, True, None

def exploit(r: RequestData, Spaces, TLog):
    ctx = context_hash(r)
    s = Spaces.get(ctx, None)
    if not s:
        from app.misc import NoSpaceError
        return r.price, NoSpaceError()
    rec, err = s.exploit(r.floor_price, r.price)
    if err:
        TLog.error(f"exploit ctx: {ctx} error: {err}")
        return r.price, err
    return rec, None

def validate(r: RequestData):
    if r.floor_price <= 0:
        return False
    if r.price <= 0:
        return False
    if r.floor_price > r.price:
        return False
    return True

import hashlib
def fnv64a(data, h=0xcbf29ce484222325):
    fnv_prime = 0x100000001b3
    for b in data:
        h ^= b
        h *= fnv_prime
        h &= 0xffffffffffffffff
    return h

def context_hash(r: RequestData):
    # Формируем данные в том же порядке, что и в Go:
    s = (r.data_center + r.bundle_id + r.tag_id + r.device_geo_country + r.ext_ad_format + r.ext_ad_format).encode('utf-8')
    h = fnv64a(s)
    # Возвращаем строку, аналогично strconv.FormatUint(h,10) в Go:
    return str(h)

def update(fb: FeedBackRequest, Spaces, Cache_, TLog):
    val, ok = Cache_.pop(fb.id)
    if not ok:
        return FeedBackResponse(ack=False)
    cctx = val["ContextHash"]
    s = Spaces.get(cctx, None)
    if not s:
        return FeedBackResponse(ack=False)
    s.update(val["Data"], fb.impression)
    return FeedBackResponse(ack=True)

def prepareSpace(ctx, Spaces):
    s = Spaces.get(ctx, None)
    if not s:
        return {"level":[]}
    return s.wc()
