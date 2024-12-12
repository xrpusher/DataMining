import json
import math
import random
import threading
import time
import numpy as np
import lightgbm as lgb
import os
from app.misc import UnfeasiblePriceError

class Bucket:
    def __init__(self, lhs, rhs, size, discount):
        self.Lhs = lhs
        self.Rhs = rhs
        self.Alpha = 1.0
        self.Beta = 1.0
        self.Buffer = []
        self.Size = size
        self.Discount = discount
        self.Pr = 0.5
        self.UpdateQty = 0

    def update(self, impression: bool):
        self.UpdateQty += 1
        if len(self.Buffer) >= self.Size:
            old = self.Buffer[-1]
            if old:
                self.Alpha -= 1
            else:
                self.Beta -= 1
            self.Buffer.pop()
        self.Buffer.insert(0, impression)
        if impression:
            self.Alpha += 1
        else:
            self.Beta += 1
        self.Pr = self.Discount*self.Pr + (1-self.Discount)*(self.Alpha/(self.Alpha+self.Beta))

class Level:
    def __init__(self, buckets):
        self.Buckets = buckets
        self.WinningCurve = [0.5 for _ in buckets]

    def exploit(self, floor_price, price):
        left, right = -1, -1
        min_ = self.Buckets[0].Lhs
        max_ = self.Buckets[-1].Rhs

        for i, b in enumerate(self.Buckets):
            if floor_price > b.Lhs and floor_price < b.Rhs:
                left = i
            if price > b.Lhs and price < b.Rhs:
                right = i

        if left == -1 or right == -1 or left > right:
            return price, UnfeasiblePriceError(price, min_, max_)

        max_val = -1e9
        recommendation = price
        for i in range(left, right+1):
            midp = self.Buckets[i].Lhs+(self.Buckets[i].Rhs-self.Buckets[i].Lhs)/2.0
            val = (price-midp)*self.WinningCurve[i]
            if val > max_val:
                max_val = val
                recommendation = midp
        return recommendation, None

    def sampleBuckets(self, price):
        for i,b in enumerate(self.Buckets):
            if price>=b.Lhs and price<=b.Rhs:
                return i
        return -1

class ExploreData:
    def __init__(self, context_hash, buckets, started):
        self.ContextHash = context_hash
        self.Buckets = buckets
        self.started = started

class TTL:
    def __init__(self):
        self.recent = []
        self.mutex = threading.Lock()
        self.size = 10
    def add(self, d):
        with self.mutex:
            self.recent.insert(0,d)
            if len(self.recent)>self.size:
                self.recent.pop()
    def time(self):
        with self.mutex:
            if len(self.recent)==0:
                return 1.0
            return 2.0*(sum(self.recent)/len(self.recent))

class UniformFlat:
    def __init__(self, context, min_price, max_price, nBins, desiredSpeed, logger):
        self.context = context
        self.desiredSpeed = desiredSpeed
        self.log = logger
        self.mutex = threading.Lock()
        self.bins = [min_price + i*(max_price-min_price)/nBins for i in range(nBins+1)]
        self.lastExplored = [time.time() for _ in range(nBins)]
        self.log.debug(f"UniformFlat {self.context}: bins initialized = {self.bins}, desiredSpeed={self.desiredSpeed}")
        
    def call(self, floorPrice, price):
        l, lok = self.findLeftmost(floorPrice)
        r, rok = self.findLeftmost(price)
        if not (lok and rok):
            return 0.0, False, UnfeasiblePriceError(price, self.bins[0], self.bins[-1])
        if r-l<2:
            return 0.0, False, None
        bin_, ok, err = self.sampleBin(l,r)
        if err:
            return 0.0, False, err
        if not ok:
            return 0.0, False, None
        new_price = self.sampleNewPrice(bin_)
        return new_price, True, None

    def findLeftmost(self, price):
        if price<=self.bins[0] or price>self.bins[-1]:
            self.log.error(f"unfeasible price: {price} [{self.bins[0]}, {self.bins[-1]}]")
            return 0, False
        idx = 0
        for i in range(len(self.bins)-1):
            if self.bins[i]<price<=self.bins[i+1]:
                idx=i
                break
        return idx, True

    def sampleBin(self, l,r):
        with self.mutex:
            candidates = []
            for i in range(l+1,r):
                t = time.time()-self.lastExplored[i]
                est_speed = 1.0/t
                candidates.append((i,est_speed))
            candidates.sort(key=lambda x:x[1])
            for (bin_,sp) in candidates:
                if random.random()<self.desiredSpeed/sp:
                    self.lastExplored[bin_] = time.time()
                    return bin_, True, None
            return 0, False, None

    def sampleNewPrice(self, bin_):
        l = self.bins[bin_]
        r = self.bins[bin_+1]
        return l+(r-l)*random.random()


class Space:
    def __init__(self, contextHash, minPrice, maxPrice, cfg, logger):
        self.ContextHash = contextHash
        self.minPrice = minPrice
        self.maxPrice = maxPrice
        self.cfg = cfg
        self.log = logger
        self.Levels = self.newLevels(minPrice,maxPrice,cfg)
        self.ExplorationQty = 0
        self.LastUpdateQty = 0
        self.ttl = TTL()
        self.mutex = threading.Lock()
        self.wcMutex = threading.Lock()
        self.explorationAlgorithm = UniformFlat(contextHash, minPrice, maxPrice, 2*cfg.bucket_size, cfg.desired_exploration_speed, logger)
        threading.Thread(target=self.background_loop, daemon=True).start()

    def background_loop(self):
        self.log.debug(f"Starting background loop for space {self.ContextHash}")
        while True:
            time.sleep(10)  
            # Проверяем состояние
            diff = self.ExplorationQty - self.LastUpdateQty
            self.log.debug(f"Space {self.ContextHash}: ExplorationQty={self.ExplorationQty}, LastUpdateQty={self.LastUpdateQty}, diff={diff}")
            if diff >= 40:
                self.log.info(f"Space {self.ContextHash}: Starting Learn() due to diff={diff}")
                self.Learn()
                self.LastUpdateQty = self.ExplorationQty

    def newLevels(self, minP,maxP,cfg):
        lambdas = self.linspace(cfg.level_size)
        levels = []
        for lam in lambdas:
            buckets = self.newBuckets(lam, minP, maxP, cfg)
            levels.append(Level(buckets))
        return levels

    def linspace(self, n):
        lam_min=0.1
        lam_max=1.8
        if n==1:
            return [lam_min]
        step = (lam_max-lam_min)/(n-1)
        return [lam_min+i*step for i in range(n)]

    def newBuckets(self, lambda_, minP,maxP,cfg):
        # генерируем экспоненциально распределенные точки и масштабируем
        bounds = self.generateBucketBounds(lambda_, minP, maxP, cfg.bucket_size)
        res = []
        for i in range(cfg.bucket_size):
            res.append(Bucket(bounds[i], bounds[i+1], cfg.buffer_size, cfg.discount))
        return res

    def generateBucketBounds(self, lam, minP,maxP,n):
        arr = [random.expovariate(lam) for _ in range(n+1)]
        arr.sort()
        arr_min = arr[0]
        arr_max = arr[-1]
        for i in range(len(arr)):
            arr[i] = ((arr[i]-arr_min)/(arr_max - arr_min))*(maxP-minP)+minP
        return arr

    def explore(self, floorPrice, price):
        self.log.debug(f"Space {self.ContextHash}: explore called with floor={floorPrice}, price={price}")
        start = time.time()
        newPrice,OK,err = self.explorationAlgorithm.call(floorPrice, price)
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
        res=[]
        for lvl in self.Levels:
            bID = lvl.sampleBuckets(price)
            res.append(bID)
        return res

    def update(self, data: ExploreData, impression: bool):
        with self.mutex:
            self.log.debug(f"update: ctx: {data.ContextHash} imp: {impression}")
            for i,bid in enumerate(data.Buckets):
                if bid==-1:
                    continue
                self.Levels[i].Buckets[bid].update(impression)
            if impression:
                self.log.debug(f"ack time {time.time()-data.started}")
                self.ttl.add(time.time()-data.started)

    def exploit(self, floorPrice, price):
        self.log.debug(f"Space {self.ContextHash}: exploit called with floor={floorPrice}, price={price}")
        with self.wcMutex:
            successes = 0
            sum_ = 0.0
            lastErr = None
            for lvl in self.Levels:
                rec, err = lvl.exploit(floorPrice, price)
                if err:
                    lastErr=err
                else:
                    sum_+=rec
                    successes+=1
            if 2*successes < len(self.Levels):
                self.log.error(f"Space {self.ContextHash}: failed to exploit #successes={successes} out of {len(self.Levels)}")
                return price, lastErr
            val = sum_/float(successes)
            if val > price:
                val = price
            self.log.info(f"Space {self.ContextHash}: exploit success, recommended_price={val}")
            return val, None



    def wc(self):
        with self.wcMutex:
            level_data = []
            for lvl in self.Levels:
                prices = []
                prs = []
                for i,b in enumerate(lvl.Buckets):
                    p = b.Lhs+(b.Rhs-b.Lhs)/2.0
                    prices.append(p)
                    prs.append(lvl.WinningCurve[i])
                level_data.append({"price":prices,"pr":prs})
            return {"level":level_data}

    def Learn(self):
            self.log.info(f"Space {self.ContextHash}: Start learning winning curve...")
            estimations_list=[]
            with self.mutex:
                for lvl in self.Levels:
                    arr=[]
                    for b,w in zip(lvl.Buckets, lvl.WinningCurve):
                        midp = b.Lhs+(b.Rhs-b.Lhs)/2.0
                        arr.append((midp,b.Pr))
                    estimations_list.append(arr)

            self.log.debug(f"Space {self.ContextHash}: Estimations before training: {estimations_list}")
            for i,estimation in enumerate(estimations_list):
                dfX = np.array([e[0] for e in estimation]).reshape(-1,1)
                dfy = np.array([e[1] for e in estimation])
                train_data = lgb.Dataset(dfX, label=dfy)
                params = {
                    "objective": "regression",
                    "metric": "l2",
                    "verbose": -1,
                    "num_leaves": 100,          # больше листьев даст более сложную модель
                    "min_child_samples": 5,     # меньше минимальных сэмплов на лист для большей вариативности
                    "learning_rate": 0.01,      # более мелкий шаг, модель станет более гибкой
                    "feature_fraction": 0.8,    # случайный подвыбор признаков
                    "bagging_fraction": 0.8,    # случайный подвыбор строк для большего шума
                    "bagging_freq": 1
                }


                model = lgb.train(params, train_data, num_boost_round=200)
                pred = model.predict(dfX)
                #pred = np.sort(pred) 
                with self.wcMutex:
                    for j in range(len(self.Levels[i].Buckets)):
                        self.Levels[i].WinningCurve[j]=pred[j]

            self.log.info(f"Space {self.ContextHash}: Learning done.")
def load_spaces(cfg, logger):
    base_dir = os.path.dirname(__file__)  # путь к space.py
    rel_path = os.path.join(base_dir, "data", "spaces_desc.json")
    abs_path = cfg.space_desc_file if cfg.space_desc_file else rel_path

    logger.debug(f"Путь к spaces_desc.json: {abs_path}")

    if not os.path.exists(abs_path):
        logger.error(f"Файл spaces_desc.json не найден: {abs_path}")
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

    logger.info(f"Описание пространств успешно загружено #{len(spaces)}")
    return spaces

