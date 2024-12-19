import json
import math
import random
import threading
import time
import numpy as np
import lightgbm as lgb
import os
from app.misc import UnfeasiblePriceError
from app.metrics import chernoff_bounds  # Импортируем нашу функцию


# Represents a single bucket within a level
class Bucket:
    def __init__(self, lhs, rhs, size, discount):
        self.Lhs = lhs  # Left-hand side boundary of the bucket
        self.Rhs = rhs  # Right-hand side boundary of the bucket
        self.Alpha = 1.0  # Parameter for tracking impressions
        self.Beta = 1.0  # Parameter for tracking non-impressions
        self.Buffer = []  # Buffer to store recent impressions
        self.Size = size  # Maximum buffer size
        self.Discount = discount  # Discount factor for exponential decay
        self.Pr = 0.5  # Initial probability
        self.UpdateQty = 0  # Count of updates

    # Updates the bucket with a new impression or non-impression
    def update(self, impression: bool):
        self.UpdateQty += 1
        if len(self.Buffer) >= self.Size:  # Remove the oldest entry if the buffer is full
            old = self.Buffer[-1]
            if old:
                self.Alpha -= 1
            else:
                self.Beta -= 1
            self.Buffer.pop()
        self.Buffer.insert(0, impression)  # Add the new impression at the start
        if impression:
            self.Alpha += 1
        else:
            self.Beta += 1
        # Update the probability with exponential smoothing
        self.Pr = self.Discount * self.Pr + (1 - self.Discount) * (self.Alpha / (self.Alpha + self.Beta))

# Represents a level containing multiple buckets
class Level:
    def __init__(self, buckets):
        self.Buckets = buckets  # List of buckets in this level
        self.WinningCurve = [0.5 for _ in buckets]  # Initialize the winning curve probabilities


    def exploit(self, floor_price, price):
            # Изменяем exploit так, чтобы он возвращал не только recommendation, но и confidence
            left, right = -1, -1
            min_ = self.Buckets[0].Lhs
            max_ = self.Buckets[-1].Rhs

            for i, b in enumerate(self.Buckets):
                if floor_price > b.Lhs and floor_price < b.Rhs:
                    left = i
                if price > b.Lhs and price < b.Rhs:
                    right = i

            if left == -1 or right == -1 or left > right:
                return price, 0.0, UnfeasiblePriceError(price, min_, max_)

            max_val = -1e9
            recommendation = price
            confidence = 0.0  # будем считать confidence как сумму (Alpha+Beta) для выбранного бакета
            for i in range(left, right + 1):
                midp = self.Buckets[i].Lhs + (self.Buckets[i].Rhs - self.Buckets[i].Lhs) / 2.0
                val = (price - midp) * self.WinningCurve[i]
                if val > max_val:
                    max_val = val
                    recommendation = midp
                    # Для confidence возьмем (Alpha+Beta) из соответствующего бакета
                    confidence = self.Buckets[i].Alpha + self.Buckets[i].Beta

            return recommendation, confidence, None



    def sampleBuckets(self, price):
            for i, b in enumerate(self.Buckets):
                if price >= b.Lhs and price <= b.Rhs:
                    return i
            return -1

class ExploreData:
    def __init__(self, context_hash, buckets, started):
        self.ContextHash = context_hash
        self.Buckets = buckets
        self.started = started
        self.is_exploit = False  # добавим флаг для отличия

# Новая структура для ExploitData
class ExploitData:
    def __init__(self, context_hash, buckets, started):
        self.ContextHash = context_hash
        self.Buckets = buckets
        self.started = started
        self.is_exploit = True

# Time-to-live tracker for recent exploration operations
class TTL:
    def __init__(self):
        self.recent = []  # List of recent exploration times
        self.mutex = threading.Lock()  # Mutex for thread safety
        self.size = 10  # Maximum number of recent times to store

    # Add a new exploration time
    def add(self, d):
        with self.mutex:
            self.recent.insert(0, d)
            if len(self.recent) > self.size:
                self.recent.pop()

    # Calculate the average time-to-live
    def time(self):
        with self.mutex:
            if len(self.recent) == 0:
                return 1.0
            return 2.0 * (sum(self.recent) / len(self.recent))

# Implements a uniform exploration strategy
class UniformFlat:
    def __init__(self, context, min_price, max_price, nBins, desiredSpeed, logger):
        self.context = context  # Context hash
        self.desiredSpeed = desiredSpeed  # Desired exploration speed
        self.log = logger  # Logger instance
        self.mutex = threading.Lock()  # Mutex for thread safety
        # Define bin boundaries for prices
        self.bins = [min_price + i * (max_price - min_price) / nBins for i in range(nBins + 1)]
        self.lastExplored = [time.time() for _ in range(nBins)]  # Timestamps of last explorations
        self.log.debug(f"UniformFlat {self.context}: bins initialized = {self.bins}, desiredSpeed={self.desiredSpeed}")

    # Main function to explore within a given range
    def call(self, floorPrice, price):
        l, lok = self.findLeftmost(floorPrice)  # Find the leftmost bin for the floor price
        r, rok = self.findLeftmost(price)  # Find the leftmost bin for the given price
        if not (lok and rok):  # Return error if the bins are invalid
            return 0.0, False, UnfeasiblePriceError(price, self.bins[0], self.bins[-1])
        if r - l < 2:  # Ensure at least one valid exploration bin exists
            return 0.0, False, None
        bin_, ok, err = self.sampleBin(l, r)  # Sample a bin within the range
        if err:
            return 0.0, False, err
        if not ok:
            return 0.0, False, None
        new_price = self.sampleNewPrice(bin_)  # Generate a new price within the sampled bin
        return new_price, True, None

    # Find the leftmost bin that contains the given price
    def findLeftmost(self, price):
        if price <= self.bins[0] or price > self.bins[-1]:
            self.log.error(f"unfeasible price: {price} [{self.bins[0]}, {self.bins[-1]}]")
            return 0, False
        idx = 0
        for i in range(len(self.bins) - 1):
            if self.bins[i] < price <= self.bins[i + 1]:
                idx = i
                break
        return idx, True

    # Select a bin to explore based on exploration speed
    def sampleBin(self, l, r):
        with self.mutex:
            candidates = []
            for i in range(l + 1, r):
                t = time.time() - self.lastExplored[i]
                est_speed = 1.0 / t
                candidates.append((i, est_speed))
            candidates.sort(key=lambda x: x[1])  # Sort bins by estimated speed
            for (bin_, sp) in candidates:
                if random.random() < self.desiredSpeed / sp:
                    self.lastExplored[bin_] = time.time()  # Update exploration timestamp
                    return bin_, True, None
            return 0, False, None

    # Generate a random price within the given bin
    def sampleNewPrice(self, bin_):
        l = self.bins[bin_]
        r = self.bins[bin_ + 1]
        return l + (r - l) * random.random()  # Uniformly sample within the bin

class Space:
    def __init__(self, contextHash, minPrice, maxPrice, cfg, logger):
        self.ContextHash = contextHash
        self.minPrice = minPrice
        self.maxPrice = maxPrice
        self.cfg = cfg
        self.log = logger
        self.Levels = self.newLevels(minPrice, maxPrice, cfg)
        self.ExplorationQty = 0
        self.LastUpdateQty = 0
        self.ttl = TTL()
        self.mutex = threading.Lock()
        self.wcMutex = threading.Lock()
        self.explorationAlgorithm = UniformFlat(contextHash, minPrice, maxPrice, 2 * cfg.bucket_size, cfg.desired_exploration_speed, logger)

        # Счетчики для online evaluation exploitation
        self.exploit_trials = 0
        self.exploit_successes = 0

        threading.Thread(target=self.background_loop, daemon=True).start()

    def background_loop(self):
        # Без изменений в логике
        while True:
            time.sleep(10)
            diff = self.ExplorationQty - self.LastUpdateQty
            if diff >= 10:
                self.Learn()
                self.LastUpdateQty = self.ExplorationQty

    def newLevels(self, minP, maxP, cfg):
        lambdas = self.linspace(cfg.level_size)
        levels = []
        for lam in lambdas:
            buckets = self.newBuckets(lam, minP, maxP, cfg)
            levels.append(Level(buckets))
        return levels

    def linspace(self, n):
        lam_min = 0.1
        lam_max = 1.8
        if n == 1:
            return [lam_min]
        step = (lam_max - lam_min) / (n - 1)
        return [lam_min + i * step for i in range(n)]

    def newBuckets(self, lambda_, minP, maxP, cfg):
        bounds = self.generateBucketBounds(lambda_, minP, maxP, cfg.bucket_size)
        res = []
        for i in range(cfg.bucket_size):
            res.append(Bucket(bounds[i], bounds[i + 1], cfg.buffer_size, cfg.discount))
        return res

    def generateBucketBounds(self, lam, minP, maxP, n):
        arr = [random.expovariate(lam) for _ in range(n + 1)]
        arr.sort()
        arr_min = arr[0]
        arr_max = arr[-1]
        for i in range(len(arr)):
            arr[i] = ((arr[i] - arr_min) / (arr_max - arr_min)) * (maxP - minP) + minP
        return arr

    def explore(self, floorPrice, price):
        self.log.debug(f"Space {self.ContextHash}: explore called with floor={floorPrice}, price={price}")
        start = time.time()
        newPrice, OK, err = self.explorationAlgorithm.call(floorPrice, price)
        if err:
            self.log.error(f"Space {self.ContextHash}: explore error: {err}")
            return 0.0, None, 0, False, err
        if not OK:
            self.log.debug(f"Space {self.ContextHash}: no exploration happened for price={price}")
            return 0.0, None, 0, False, None
        buckets = self.sampleBuckets(newPrice)
        data = ExploreData(self.ContextHash, buckets, start)
        with self.mutex:
            self.ExplorationQty += 1
        self.log.info(f"Space {self.ContextHash}: exploration success, newPrice={newPrice}, ExplorationQty={self.ExplorationQty}")
        return newPrice, data, self.ttl.time(), True, None

    def sampleBuckets(self, price):
        res = []
        for lvl in self.Levels:
            bID = lvl.sampleBuckets(price)
            res.append(bID)
        return res

    def update(self, data, impression: bool):
        # data может быть ExploreData или ExploitData
        with self.mutex:
            self.log.debug(f"update: ctx: {data.ContextHash} imp: {impression} exploit={data.is_exploit}")
            for i, bid in enumerate(data.Buckets):
                if bid == -1:
                    continue
                self.Levels[i].Buckets[bid].update(impression)
            if impression and not data.is_exploit:
                # Только для exploration считаем TTL
                self.log.debug(f"ack time {time.time() - data.started}")
                self.ttl.add(time.time() - data.started)

            # Если это feedback от exploitation:
            if data.is_exploit:
                self.exploit_trials += 1
                if impression:
                    self.exploit_successes += 1

    def exploit(self, floorPrice, price):
        self.log.debug(f"Space {self.ContextHash}: exploit called with floor={floorPrice}, price={price}")
        with self.wcMutex:
            weighted_sum = 0.0
            total_confidence = 0.0
            lastErr = None
            for lvl in self.Levels:
                rec, conf, err = lvl.exploit(floorPrice, price)
                if err:
                    lastErr = err
                else:
                    weighted_sum += rec * conf
                    total_confidence += conf
            if total_confidence == 0:
                self.log.error(f"Space {self.ContextHash}: failed to exploit, no confidence")
                return price, lastErr
            val = weighted_sum / total_confidence
            if val > price:
                val = price

            # Сохраним ExploitData в кеш позже через handlers
            self.log.info(f"Space {self.ContextHash}: exploit success, recommended_price={val}")
            buckets = self.sampleBuckets(val)
            data = ExploitData(self.ContextHash, buckets, time.time())
            return val, data, None

    def wc(self):
        with self.wcMutex:
            level_data = []
            for lvl in self.Levels:
                prices = []
                prs = []
                for i, b in enumerate(lvl.Buckets):
                    p = b.Lhs + (b.Rhs - b.Lhs) / 2.0
                    prices.append(p)
                    prs.append(lvl.WinningCurve[i])
                level_data.append({"price": prices, "pr": prs})
            return {"level": level_data}

    def Learn(self):
        self.log.info(f"Space {self.ContextHash}: Start learning winning curve...")
        estimations_list = []
        with self.mutex:
            for lvl in self.Levels:
                arr = []
                for b, w in zip(lvl.Buckets, lvl.WinningCurve):
                    midp = b.Lhs + (b.Rhs - b.Lhs) / 2.0
                    arr.append((midp, b.Pr))
                estimations_list.append(arr)

        for i, estimation in enumerate(estimations_list):
            dfX = np.array([e[0] for e in estimation]).reshape(-1, 1)
            dfy = np.array([e[1] for e in estimation])
            train_data = lgb.Dataset(dfX, label=dfy)
            params = {
                "objective": "regression",
                "metric": "l2",
                "verbose": -1,
                "num_leaves": 100,
                "min_child_samples": 5,
                "learning_rate": 0.01,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1
            }
            model = lgb.train(params, train_data, num_boost_round=200)
            pred = model.predict(dfX)
            with self.wcMutex:
                for j in range(len(self.Levels[i].Buckets)):
                    self.Levels[i].WinningCurve[j] = pred[j]
        self.log.info(f"Space {self.ContextHash}: Learning done.")

    def evaluate_online(self):
        # Возвращаем метрики exploitation
        with self.mutex:
            if self.exploit_trials == 0:
                return {"exploit_ctr": None, "interval": None}
            p = self.exploit_successes / self.exploit_trials
            low, high = chernoff_bounds(p, self.exploit_trials, delta=0.05)
            return {
                "exploit_ctr": p,
                "interval": [low, high],
                "trials": self.exploit_trials
            }

def load_spaces(cfg, logger):
    base_dir = os.path.dirname(__file__)
    rel_path = os.path.join(base_dir, "data", "spaces_desc.json")
    abs_path = cfg.space_desc_file if cfg.space_desc_file else rel_path

    logger.debug(f"Path to spaces_desc.json: {abs_path}")

    if not os.path.exists(abs_path):
        logger.error(f"File spaces_desc.json not found: {abs_path}")
        return {}

    with open(abs_path, 'r') as f:
        spaces_desc = json.load(f)

    spaces = {}
    for s in spaces_desc:
        context_hash = s["context_hash"]
        min_price = s["min_price"]
        max_price = s["max_price"]
        sp = Space(context_hash, min_price, max_price, cfg, logger)
        spaces[context_hash] = sp

    logger.info(f"Successfully loaded spaces #{len(spaces)}")
    return spaces