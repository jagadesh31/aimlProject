"""Project configuration for Continuous SER + Temporal Emotion Tracking."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MELD_DIR = DATA_DIR / "meld"
RAW_DIR = MELD_DIR / "raw"
LABEL_DIR = MELD_DIR / "labels"
VIDEO_DIR = MELD_DIR / "video"
AUDIO_DIR = MELD_DIR / "audio"
CACHE_DIR = DATA_DIR / "cache"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
TIMELINE_DIR = OUTPUT_DIR / "timelines"
DEMO_DIR = OUTPUT_DIR / "demo"
DOCS_DIR = ROOT / "docs"

SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
MIN_DURATION = 1.5  # pad only very short clips so pooling still has time steps
MAX_DURATION = 6.0  # crop longer utterances; do not pad every clip out to 6s
WIN_SEC = 2.0
HOP_SEC = 1.0

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2EMO = {i: e for e, i in EMO2IDX.items()}
NUM_CLASSES = len(EMOTIONS)

BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 4
NUM_WORKERS = 0  # Windows-friendly
SEED = 42
DEVICE = "cuda"  # falls back to cpu in train script

# Temporal tracking defaults
MIN_CONF = 0.35
MIN_STABLE_SEC = 1.0
TRANS_SEC = 0.5
SMOOTH_WINDOW = 3
