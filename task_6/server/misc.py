class NoSpaceError(Exception):
    def __str__(self):
        return "no space"

class ValidationError(Exception):
    def __str__(self):
        return "validation error"

class UnfeasiblePriceError(Exception):
    def __init__(self, price: float, min_val: float, max_val: float):
        self.price = price
        self.min_val = min_val
        self.max_val = max_val
    def __str__(self):
        return f"unfeasible price {self.price} [{self.min_val}, {self.max_val}]"

class Config:
    BUFFER_SIZE: int
    DISCOUNT: float
    DESIRED_EXPLORATION_SPEED: float
    LOG_LEVEL: str
    LEVEL_SIZE: int
    BUCKET_SIZE: int
    SPACE_DESC_FILE: str
