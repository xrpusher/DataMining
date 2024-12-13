import time
import threading

# Custom error indicating a validation issue.
class ValidationError(Exception):
    def __str__(self):
        return "validation error"

# Custom error indicating that no space is available for a given context.
class NoSpaceError(Exception):
    def __str__(self):
        return "no space"

# Custom error indicating that a given price is not feasible within bounds.
class UnfeasiblePriceError(Exception):
    def __init__(self, price, min_, max_):
        self.price = price       # The problematic price
        self.min_ = min_         # Minimum allowable price
        self.max_ = max_         # Maximum allowable price
    def __str__(self):
        return f"unfeasible price {self.price} [{self.min_}, {self.max_}]"

# Class representing an individual cache item with a value, expiration time, and callback.
class CacheItem:
    def __init__(self, value, expire, cb):
        self.value = value       # Cached value
        self.expire = expire     # Expiration time for the cache
        self.cb = cb             # Callback to execute when the cache expires

    # Method to check if the cache item has expired.
    def is_expired(self):
        return time.time() > self.expire

# Simple cache implementation with time-to-live (TTL) functionality.
class Cache:
    def __init__(self, ttl_seconds=1, logger=None):
        self.items = {}                  # Dictionary to store cache items
        self.ttl = ttl_seconds           # Default TTL for cache items
        self.logger = logger             # Logger for debugging
        self.mu = threading.Lock()       # Mutex for thread-safe operations
        # Background thread to clean up expired cache items.
        self._cleaner_thread = threading.Thread(target=self.cleaner, daemon=True)
        self._cleaner_thread.start()

    # Add a new item to the cache with an optional TTL and callback.
    def set(self, key, value, ttl=None, cb=lambda x: True):
        if ttl is None:
            ttl = self.ttl
        expire = time.time() + ttl
        with self.mu:
            self.items[key] = CacheItem(value, expire, cb)

    # Retrieve an item from the cache by key.
    def get(self, key):
        with self.mu:
            item = self.items.get(key)
            if not item:
                return None, False
            if item.is_expired():
                # Remove expired item from the cache.
                del self.items[key]
                return None, False
            return item.value, True

    # Remove an item from the cache and return it.
    def pop(self, key):
        with self.mu:
            item = self.items.get(key)
            if not item:
                return None, False
            del self.items[key]
            if item.is_expired():
                return item.value, False
            return item.value, True

    # Background thread function to clean up expired cache items.
    def cleaner(self):
        while True:
            time.sleep(0.1)
            now = time.time()
            rm = []
            with self.mu:
                for k, v in list(self.items.items()):
                    if v.is_expired():
                        v.cb(v.value)
                        rm.append(k)
                for k in rm:
                    del self.items[k]

# Class for tracking recent events and calculating average times.
class TTL:
    def __init__(self):
        self.recent = []            # List of recent event durations
        self.size = 10              # Maximum number of events to track
        self.mu = threading.Lock()  # Mutex for thread-safe operations

    # Add a new duration to the recent events list.
    def add(self, d):
        with self.mu:
            self.recent.insert(0, d)
            if len(self.recent) > self.size:
                self.recent.pop()

    # Calculate the average time of recent events, scaled by a factor of 2.
    def time(self):
        with self.mu:
            if len(self.recent) == 0:
                return 1.0
            return 2.0 * (sum(self.recent) / len(self.recent))