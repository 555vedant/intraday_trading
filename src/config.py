# src/config.py

DATA_PATH = "data/nifty_intraday.csv"

FEATURE_COLUMNS = [
    "open", "high", "low", "close",
    "return", "hl_range", "co_diff",
    "sma_5", "sma_10"
]

TARGET_COLUMN = "target"
TRAIN_RATIO = 0.7
RANDOM_STATE = 42

PROBA_THRESHOLD = 0.50

