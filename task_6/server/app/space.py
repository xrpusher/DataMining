import json
import math
import random
import threading
import time
import numpy as np
import lightgbm as lgb
import os
from app.misc import UnfeasiblePriceError

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

    # Find the optimal recommendation based on the winning curve
    def exploit(self, floor_price, price):
        left, right = -1, -1
        min_ = self.Buckets[0].Lhs  # Minimum boundary
        max_ = self.Buckets[-1].Rhs  # Maximum boundary

        # Identify the range of buckets affected by the floor price and given price
        for i, b in enumerate(self.Buckets):
            if floor_price > b.Lhs and floor_price < b.Rhs:
                left = i
            if price > b.Lhs and price < b.Rhs:
                right = i

        # Return an error if no valid buckets are found
        if left == -1 or right == -1 or left > right:
            return price, UnfeasiblePriceError(price, min_, max_)

        max_val = -1e9
        recommendation = price
        # Iterate through the relevant buckets and find the optimal midpoint
        for i in range(left, right + 1):
            midp = self.Buckets[i].Lhs + (self.Buckets[i].Rhs - self.Buckets[i].Lhs) / 2.0
            val = (price - midp) * self.WinningCurve[i]
            if val > max_val:
                max_val = val
                recommendation = midp
        return recommendation, None

    # Find the index of the bucket that contains the given price
    def sampleBuckets(self, price):
        for i, b in enumerate(self.Buckets):
            if price >= b.Lhs and price <= b.Rhs:
                return i
        return -1

# Represents data for an exploration operation
class ExploreData:
    def __init__(self, context_hash, buckets, started):
        self.ContextHash = context_hash  # Hash identifying the context
        self.Buckets = buckets  # List of bucket indices
        self.started = started  # Start time of the exploration

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
    # Constructor to initialize a Space instance
    def __init__(self, contextHash, minPrice, maxPrice, cfg, logger):
        self.ContextHash = contextHash  # Unique hash for the space
        self.minPrice = minPrice  # Minimum price for the space
        self.maxPrice = maxPrice  # Maximum price for the space
        self.cfg = cfg  # Configuration object containing parameters
        self.log = logger  # Logger instance for logging
        self.Levels = self.newLevels(minPrice, maxPrice, cfg)  # Create levels with buckets
        self.ExplorationQty = 0  # Tracks exploration attempts
        self.LastUpdateQty = 0  # Tracks the last update count
        self.ttl = TTL()  # Time-to-live tracking for exploration data
        self.mutex = threading.Lock()  # Mutex for thread-safe operations
        self.wcMutex = threading.Lock()  # Mutex for WinningCurve updates
        self.explorationAlgorithm = UniformFlat(contextHash, minPrice, maxPrice, 2 * cfg.bucket_size, cfg.desired_exploration_speed, logger)  # Exploration algorithm
        # Start a background thread for monitoring and learning
        threading.Thread(target=self.background_loop, daemon=True).start()

    # Background loop for periodic updates
    def background_loop(self):
        self.log.debug(f"Starting background loop for space {self.ContextHash}")
        while True:
            time.sleep(10)  # Sleep for 10 seconds
            # Check the difference in exploration quantity
            diff = self.ExplorationQty - self.LastUpdateQty
            self.log.debug(f"Space {self.ContextHash}: ExplorationQty={self.ExplorationQty}, LastUpdateQty={self.LastUpdateQty}, diff={diff}")
            if diff >= 40:  # Trigger learning if sufficient exploration has occurred
                self.log.info(f"Space {self.ContextHash}: Starting Learn() due to diff={diff}")
                self.Learn()
                self.LastUpdateQty = self.ExplorationQty  # Update the last processed quantity

    # Create levels with corresponding buckets
    def newLevels(self, minP, maxP, cfg):
        lambdas = self.linspace(cfg.level_size)  # Generate lambda values for levels
        levels = []
        for lam in lambdas:
            buckets = self.newBuckets(lam, minP, maxP, cfg)  # Create buckets for each level
            levels.append(Level(buckets))
        return levels

    # Generate evenly spaced lambda values
    def linspace(self, n):
        lam_min = 0.1
        lam_max = 1.8
        if n == 1:
            return [lam_min]
        step = (lam_max - lam_min) / (n - 1)
        return [lam_min + i * step for i in range(n)]

    # Create buckets for a given level
    def newBuckets(self, lambda_, minP, maxP, cfg):
        # Generate exponential bucket boundaries
        bounds = self.generateBucketBounds(lambda_, minP, maxP, cfg.bucket_size)
        res = []
        for i in range(cfg.bucket_size):
            res.append(Bucket(bounds[i], bounds[i + 1], cfg.buffer_size, cfg.discount))
        return res

    # Generate boundaries for buckets using exponential distribution
    def generateBucketBounds(self, lam, minP, maxP, n):
        arr = [random.expovariate(lam) for _ in range(n + 1)]
        arr.sort()  # Sort to maintain order
        arr_min = arr[0]
        arr_max = arr[-1]
        # Scale the boundaries to fit the price range
        for i in range(len(arr)):
            arr[i] = ((arr[i] - arr_min) / (arr_max - arr_min)) * (maxP - minP) + minP
        return arr

    # Explore potential new prices
    def explore(self, floorPrice, price):
        self.log.debug(f"Space {self.ContextHash}: explore called with floor={floorPrice}, price={price}")
        start = time.time()
        # Use the exploration algorithm to suggest a new price
        newPrice, OK, err = self.explorationAlgorithm.call(floorPrice, price)
        if err:
            self.log.error(f"Space {self.ContextHash}: explore error: {err}")
            return 0.0, None, 0, False, err
        if not OK:
            self.log.debug(f"Space {self.ContextHash}: no exploration happened for price={price}")
            return 0.0, None, 0, False, None
        buckets = self.sampleBuckets(newPrice)  # Find the relevant buckets for the new price
        data = ExploreData(self.ContextHash, buckets, start)  # Create exploration data
        with self.mutex:
            self.ExplorationQty += 1  # Increment exploration count
        self.log.info(f"Space {self.ContextHash}: exploration success, newPrice={newPrice}, ExplorationQty={self.ExplorationQty}")
        return newPrice, data, self.ttl.time(), True, None

    # Sample relevant buckets for a given price
    def sampleBuckets(self, price):
        res = []
        for lvl in self.Levels:
            bID = lvl.sampleBuckets(price)
            res.append(bID)
        return res

    # Update space data based on feedback
    def update(self, data: ExploreData, impression: bool):
        with self.mutex:
            self.log.debug(f"update: ctx: {data.ContextHash} imp: {impression}")
            for i, bid in enumerate(data.Buckets):
                if bid == -1:
                    continue
                self.Levels[i].Buckets[bid].update(impression)  # Update the corresponding bucket
            if impression:
                self.log.debug(f"ack time {time.time() - data.started}")
                self.ttl.add(time.time() - data.started)  # Update time-to-live metrics

    # Exploit existing data to recommend an optimal price
    def exploit(self, floorPrice, price):
        self.log.debug(f"Space {self.ContextHash}: exploit called with floor={floorPrice}, price={price}")
        with self.wcMutex:
            successes = 0
            sum_ = 0.0
            lastErr = None
            for lvl in self.Levels:
                rec, err = lvl.exploit(floorPrice, price)
                if err:
                    lastErr = err
                else:
                    sum_ += rec
                    successes += 1
            if 2 * successes < len(self.Levels):
                self.log.error(f"Space {self.ContextHash}: failed to exploit #successes={successes} out of {len(self.Levels)}")
                return price, lastErr
            val = sum_ / float(successes)
            if val > price:
                val = price
            self.log.info(f"Space {self.ContextHash}: exploit success, recommended_price={val}")
            return val, None

    # Retrieve the winning curve for all levels
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

    # Train and update the winning curve based on collected data
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

        self.log.debug(f"Space {self.ContextHash}: Estimations before training: {estimations_list}")
        for i, estimation in enumerate(estimations_list):
            dfX = np.array([e[0] for e in estimation]).reshape(-1, 1)
            dfy = np.array([e[1] for e in estimation])
            train_data = lgb.Dataset(dfX, label=dfy)
            # Define LightGBM parameters
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
            model = lgb.train(params, train_data, num_boost_round=200)  # Train the model
            pred = model.predict(dfX)  # Predict values
            with self.wcMutex:
                for j in range(len(self.Levels[i].Buckets)):
                    self.Levels[i].WinningCurve[j] = pred[j]  # Update the winning curve
        self.log.info(f"Space {self.ContextHash}: Learning done.")

# Load space configurations from a file
def load_spaces(cfg, logger):
    base_dir = os.path.dirname(__file__)  # Path to the current file
    rel_path = os.path.join(base_dir, "data", "spaces_desc.json")  # Relative path to spaces_desc.json
    abs_path = cfg.space_desc_file if cfg.space_desc_file else rel_path

    logger.debug(f"Path to spaces_desc.json: {abs_path}")

    if not os.path.exists(abs_path):
        logger.error(f"File spaces_desc.json not found: {abs_path}")
        return {}

    with open(abs_path, 'r') as f:
        spaces_desc = json.load(f)  # Load JSON data

    spaces = {}
    for s in spaces_desc:
        context_hash = s["context_hash"]
        min_price = s["min_price"]
        max_price = s["max_price"]
        sp = Space(context_hash, min_price, max_price, cfg, logger)  # Create Space instances
        spaces[context_hash] = sp

    logger.info(f"Successfully loaded spaces #{len(spaces)}")
    return spaces
