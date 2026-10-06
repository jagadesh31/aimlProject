"""Project configuration for Continuous SER + Temporal Emotion Tracking."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MELD_REPO = DATA_DIR / "MELD_repo"
LABEL_DIR = MELD_REPO / "data" / "MELD"
VIDEO_DIR = DATA_DIR / "meld_video"
AUDIO_DIR = DATA_DIR / "meld_audio"
CACHE_DIR = DATA_DIR / "mel_cache"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"

SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
MAX_DURATION = 6.0  # seconds; pad/crop utterances for training
WIN_SEC = 2.0
HOP_SEC = 1.0

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2EMO = {i: e for e, i in EMO2IDX.items()}
NUM_CLASSES = len(EMOTIONS)

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # Windows-friendly
SEED = 42
DEVICE = "cuda"  # falls back to cpu in train script

# Temporal tracking defaults
MIN_CONF = 0.35
MIN_STABLE_SEC = 1.0
TRANS_SEC = 0.5
SMOOTH_WINDOW = 3
