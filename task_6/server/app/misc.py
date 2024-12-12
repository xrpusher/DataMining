import time
import threading

# Ошибки
class ValidationError(Exception):
    def __str__(self):
        return "validation error"

class NoSpaceError(Exception):
    def __str__(self):
        return "no space"

class UnfeasiblePriceError(Exception):
    def __init__(self, price, min_, max_):
        self.price = price
        self.min_ = min_
        self.max_ = max_
    def __str__(self):
        return f"unfeasible price {self.price} [{self.min_}, {self.max_}]"


# Простая реализация кэша с TTL
class CacheItem:
    def __init__(self, value, expire, cb):
        self.value = value
        self.expire = expire
        self.cb = cb

    def is_expired(self):
        return time.time() > self.expire

class Cache:
    def __init__(self, ttl_seconds=1, logger=None):
        self.items = {}
        self.ttl = ttl_seconds
        self.logger = logger
        self.mu = threading.Lock()
        # Запускаем очистку
        self._cleaner_thread = threading.Thread(target=self.cleaner, daemon=True)
        self._cleaner_thread.start()

    def set(self, key, value, ttl=None, cb=lambda x: True):
        if ttl is None:
            ttl = self.ttl
        expire = time.time() + ttl
        with self.mu:
            self.items[key] = CacheItem(value, expire, cb)

    def get(self, key):
        with self.mu:
            item = self.items.get(key)
            if not item:
                return None, False
            if item.is_expired():
                # expired
                del self.items[key]
                return None, False
            return item.value, True

    def pop(self, key):
        with self.mu:
            item = self.items.get(key)
            if not item:
                return None, False
            del self.items[key]
            if item.is_expired():
                return item.value, False
            return item.value, True

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

class TTL:
    def __init__(self):
        self.recent = []
        self.size = 10
        self.mu = threading.Lock()

    def add(self, d):
        with self.mu:
            self.recent.insert(0, d)
            if len(self.recent) > self.size:
                self.recent.pop()

    def time(self):
        with self.mu:
            if len(self.recent) == 0:
                return 1.0
            return 2.0 * (sum([r for r in [x for x in self.recent]])/len(self.recent))

# Конфиги, ошибки уже определены, доп. функционал в space.py