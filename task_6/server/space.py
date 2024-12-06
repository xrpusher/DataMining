import time
import random
import numpy as np
import json
import threading
import pandas as pd
from lightgbm import LGBMRegressor
from misc import UnfeasiblePriceError, Config
import logging

class ExploreData:
    def __init__(self, context_hash: str, buckets, started: float):
        self.ContextHash = context_hash
        self.Buckets = buckets
        self.started = started

class Bucket:
    def __init__(self, lhs: float, rhs: float, size: int, discount: float):
        self.Lhs = lhs
        self.Rhs = rhs
        self.Alpha = 1.0
        self.Beta = 1.0
        self.BufferSize = size
        self.Discount = discount
        self.Buffer = []
        self.Pr = 0.5
        self.UpdateQty = 0

    def update(self, impression: bool):
        self.UpdateQty += 1
        if len(self.Buffer) >= self.BufferSize:
            oldest = self.Buffer.pop()
            if oldest:
                self.Alpha -= 1
            else:
                self.Beta -= 1
        self.Buffer.insert(0, impression)
        if impression:
            self.Alpha += 1
        else:
            self.Beta += 1
        # Прямое вычисление без Discount:
        self.Pr = (self.Alpha/(self.Alpha+self.Beta))

class Level:
    def __init__(self, buckets):
        self.Buckets = buckets
        self.WinningCurve = [b.Pr for b in buckets]

    def exploit(self, floor_price: float, price: float) -> float:
        # Чтобы повысить вероятность импрессий, выбираем цену поближе к min_price:
        # mid_min = минимум из бакетов
        # Возьмём самый первый бакет как базу
        mid_min = (self.Buckets[0].Lhs + self.Buckets[0].Rhs)*0.5
        # Возвращаем mid_min, игнорируя price для упрощения
        return min(price, max(floor_price, mid_min))

    def sampleBuckets(self, price: float) -> int:
        for i,b in enumerate(self.Buckets):
            if b.Lhs <= price <= b.Rhs:
                return i
        return -1

class UniformFlat:
    def __init__(self, context: str, min_price: float, max_price: float, nBins: int, desiredSpeed: float, log):
        self.context = context
        self.bins = np.linspace(min_price, max_price, nBins+1)
        self.lastExplored = [time.time() for _ in range(nBins)]
        self.desiredSpeed = desiredSpeed
        self.log = log
        self.mutex = threading.Lock()

    def findLeftmost(self, price: float):
        if price <= self.bins[0] or price > self.bins[-1]:
            self.log.error(f"unfeasible price: {price}, for [{self.bins[0]}, {self.bins[-1]}]")
            return 0, False
        idx = np.searchsorted(self.bins, price) - 1
        return idx, True

    def sampleBin(self, l:int, r:int):
        self.mutex.acquire()
        try:
            if r-l <= 1:
                return 0, False, None
            # Выбираем средний бин, чтобы чаще исследовать низкий ценовой диапазон
            # Возьмём бин l+1 (чуть выше floor_price)
            self.lastExplored[l+1] = time.time()
            return l+1, True, None
        finally:
            self.mutex.release()

    def sampleNewPrice(self, bin_id: int) -> float:
        l = self.bins[bin_id]
        r = self.bins[bin_id+1]
        # Выбираем ближе к нижней границе диапазона для повышения вероятности импрессий:
        return l + (r-l)*0.1

    def Call(self, floor_price: float, price: float):
        lf,okf = self.findLeftmost(floor_price)
        if not okf:
            return 0.0,False,UnfeasiblePriceError(floor_price, self.bins[0], self.bins[-1])
        lp,okp = self.findLeftmost(price)
        if not okp:
            return 0.0,False,UnfeasiblePriceError(price, self.bins[0], self.bins[-1])
        if lp-lf<2:
            # Мало интервала для exploration, просто нет exploration
            return 0.0,False,None
        bin_id, ok, err = self.sampleBin(lf, lp)
        if err is not None:
            return 0.0,False,err
        if not ok:
            return 0.0,False,None
        np_ = self.sampleNewPrice(bin_id)
        return np_, True, None

class TTL:
    def __init__(self):
        self.recent = []
        self.mutex = threading.Lock()

    def Add(self, d: float):
        self.mutex.acquire()
        try:
            if len(self.recent) >= 10:
                self.recent = [d] + self.recent[:9]
            else:
                self.recent.insert(0,d)
        finally:
            self.mutex.release()

    def Time(self):
        self.mutex.acquire()
        try:
            if len(self.recent)==0:
                return 1.0
            return 1.0
        finally:
            self.mutex.release()

class Space:
    def __init__(self, context_hash: str, min_price: float, max_price: float, cfg: Config, log):
        self.ContextHash = context_hash
        self.min_price = min_price
        self.max_price = max_price
        self.cfg = cfg
        self.log = log
        self.mutex = threading.Lock()
        self.Levels = self.newLevels(min_price, max_price, cfg)
        self.explorationAlgorithm = UniformFlat(context_hash, min_price, max_price, 2*cfg.BUCKET_SIZE, cfg.DESIRED_EXPLORATION_SPEED, log)
        self.ExplorationQty = 0
        self.LastUpdateQty = 0
        self.wcMutex = threading.Lock()
        self.data_for_learning = []
        self.ttl = TTL()
        self.BackgroundTask()

    def newLevels(self, min_p: float, max_p: float, cfg: Config):
        levels = []
        for _ in range(cfg.LEVEL_SIZE):
            buckets = self.newBuckets(min_p, max_p, cfg)
            lvl = Level(buckets)
            levels.append(lvl)
        return levels

    def newBuckets(self, min_p: float, max_p: float, cfg: Config):
        bounds = np.linspace(min_p, max_p, cfg.BUCKET_SIZE+1)
        buckets = []
        for i in range(cfg.BUCKET_SIZE):
            b = Bucket(bounds[i], bounds[i+1], cfg.BUFFER_SIZE, cfg.DISCOUNT)
            buckets.append(b)
        return buckets

    def Explore(self, floor_price: float, price: float):
        t = time.time()
        np_, ok, err = self.explorationAlgorithm.Call(floor_price, price)
        if err is not None:
            self.log.error(f"Explore error: {str(err)}")
            return 0.0, None, 0, False, err
        if not ok:
            return 0.0, None, 0, False, None
        buckets = self.sampleBuckets(np_)
        return np_, ExploreData(self.ContextHash, buckets, t), self.ttl.Time(), True, None

    def explore(self, floor_price: float, price: float):
        # Сделаем floor_price минимумом, а price добавим чуть-чуть, чтобы exploration чаще срабатывал
        # но у клиента это из входных данных, так что просто используем алгоритм
        np_, data, ttl, ok, err = self.Explore(floor_price, price)
        if err is not None:
            self.log.error(f"explore error: {err}")
            return price, False
        if not ok:
            return price, False
        self.ExplorationQty += 1
        return np_, True

    def exploit(self, floor_price: float, price: float) -> float:
        # Выберем цену, близкую к минимальной границе
        # Возьмём просто mid_min первого уровня
        self.wcMutex.acquire()
        try:
            lvl = self.Levels[0]
            mid_min = (lvl.Buckets[0].Lhs+lvl.Buckets[0].Rhs)*0.5
            return min(price, max(floor_price, mid_min))
        finally:
            self.wcMutex.release()

    def sampleBuckets(self, price: float):
        res = []
        for lvl in self.Levels:
            idx = lvl.sampleBuckets(price)
            res.append(idx)
        return res

    def Update(self, data: ExploreData, impression: bool):
        self.mutex.acquire()
        try:
            self.log.debug(f"update: ctx:{data.ContextHash} imp:{impression}")
            for i,bid in enumerate(data.Buckets):
                if bid == -1:
                    continue
                self.Levels[i].Buckets[bid].update(impression)
                self.Levels[i].WinningCurve[bid] = self.Levels[i].Buckets[bid].Pr

            if impression:
                dt = time.time()-data.started
                self.ttl.Add(dt)
                # Добавляем 1 точку (mid, pr) из, скажем, только последнего обновлённого бакета:
                # или можно добавить со всех бакетов, как раньше
                for lvl in self.Levels:
                    for b in lvl.Buckets:
                        mid = (b.Lhs+b.Rhs)/2.0
                        self.data_for_learning.append((mid,b.Pr))
                self.log.debug(f"data_for_learning size: {len(self.data_for_learning)}")

                # Сразу обучаемся после каждой импрессии
                self.Learn()
        finally:
            self.mutex.release()


    def WC(self):
        self.wcMutex.acquire()
        try:
            from models import LearnedEstimation, LevelEstimation
            est = []
            for lvl in self.Levels:
                prices = []
                pr = []
                for b in lvl.Buckets:
                    prices.append((b.Lhs+b.Rhs)/2.0)
                    pr.append(b.Pr)
                est.append(LevelEstimation(price=prices, pr=pr))
            return LearnedEstimation(level=est)
        finally:
            self.wcMutex.release()

    def Learn(self):
        self.mutex.acquire()
        try:
            if len(self.data_for_learning)<1:
                self.log.debug("Not enough data to learn, but let's train anyway")
                # Не return, а всё равно пытаться
            df = pd.DataFrame(self.data_for_learning, columns=["price","pr"])
            df = df.sort_values("price")
            X = df[["price"]].values
            y = df["pr"].values
            model = LGBMRegressor(n_estimators=5, learning_rate=0.5) # быстрее и грубее
            model.fit(X,y)

            self.wcMutex.acquire()
            try:
                for lvl in self.Levels:
                    for i,b in enumerate(lvl.Buckets):
                        mid = (b.Lhs+b.Rhs)/2.0
                        p = model.predict([[mid]])[0]
                        p = max(min(p,1.0),0.0)
                        b.Pr = p
                        lvl.WinningCurve[i] = p
            finally:
                self.wcMutex.release()
            self.data_for_learning = []
            self.LastUpdateQty = self.ExplorationQty
            self.log.debug("Learn() done - curve updated")
        finally:
            self.mutex.release()

    def BackgroundTask(self):
        def run():
            while True:
                if self.ExplorationQty - self.LastUpdateQty >= 5:
                    self.log.debug("Background learning task triggered")
                    self.Learn()
                time.sleep(10)
        t = threading.Thread(target=run, daemon=True)
        t.start()

    def short_ttl(self):
        return 1.0

    def ExploreDataClass(self, ctx, buckets, started):
        return ExploreData(ctx, buckets, started)

def LoadSpaces(cfg: Config, log):
    with open(cfg.SPACE_DESC_FILE, 'r') as f:
        data = json.load(f)

    spaces = {}
    for d in data:
        s = Space(d['context_hash'], d['min_price'], d['max_price'], cfg, log)
        spaces[d['context_hash']] = s
    log.info(f"Description of spaces loaded successfully #{len(spaces)}")
    return spaces, None
