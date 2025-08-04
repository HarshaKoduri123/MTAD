# config.py

import os
import logging

# ======= File Paths =======
DATA = r'C:\Users\remin\OneDrive\Documents\MTAD\Data\AIS_data_east_coast.csv'
VAL_DATA = r'C:\Users\remin\OneDrive\Documents\MTAD\Data\VAL_AIS_data_east_coast.csv'

SAVE_DIR = r'C:\Users\remin\OneDrive\Documents\MTAD\weights'

os.makedirs(SAVE_DIR, exist_ok=True)

# ======= Sequence Settings =======
SEQ_LENGTH = 50
SEQ_LEN_MODEL = 100  # for the transformer encoder

# ======= Feature Settings =======
REQUIRED_COLS = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading']
FEATURES = ['DeltaTime', 'DeltaLat', 'DeltaLon', 'SOG', 'COG', 'Heading']
FEATURE_DIM = len(FEATURES)

# ======= Model Settings =======
D_MODEL = 64
BATCH_SIZE = 2048
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
CHECKPOINT_INTERVAL = 10

# ======= Loss Weighting =======
MODALITY_INDICES = {
    'time': [0],
    'spatial': [1, 2],
    'speed': [3],
    'direction': [4, 5],
}

RAW_WEIGHTS = {
    'time': 1.0,
    'spatial': 3.0,
    'speed': 2.0,
    'direction': 2.0,
}
TOTAL_WEIGHT = sum(RAW_WEIGHTS.values())
LOSS_WEIGHTS = {k: v / TOTAL_WEIGHT for k, v in RAW_WEIGHTS.items()}

# ======= Logging =======
LOG_LEVEL = logging.INFO

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(LOG_LEVEL)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
