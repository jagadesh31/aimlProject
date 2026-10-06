# Continuous Speech Emotion Recognition with Temporal Emotion Tracking

Complete runnable project:
- CNN + BiLSTM window-level SER
- Temporal tracking -> stable regions + transitions -> emotion timeline

## Current status

| Item | Status |
|------|--------|
| Code pipeline | Ready |
| RAVDESS bootstrap train | Done (`checkpoints/best_model.pt`) |
| Timeline demo | Done (`outputs/timeline_demo_sequence.json`) |
| MELD full download | In progress (`data/raw/MELD.Raw.tar.gz` ~10.9 GB) |

MELD is the main conversational dataset for your problem statement.  
RAVDESS was used to finish training/demo while MELD downloads.

## Setup

```bash
pip install -r requirements.txt
# ffmpeg: winget install Gyan.FFmpeg   (or imageio-ffmpeg, already in requirements)
```

## Quick demo (already trained)

```bash
python scripts/make_demo_audio.py
python scripts/infer_timeline.py --audio outputs/demo_sequence.wav
python scripts/evaluate.py --source ravdess --split test
```

## Full MELD training (after download completes)

```bash
# if download was interrupted:
python scripts/download_meld.py

# unpack mp4 -> wav, train, evaluate, dialogue timeline
python scripts/finish_meld_pipeline.py
```

Or step-by-step:

```bash
python scripts/prepare_meld.py
python scripts/train.py --source meld --epochs 15
python scripts/evaluate.py --source meld --split test
python scripts/infer_timeline.py --split test --dialogue-id 0
```

## Project layout

```text
config.py
src/
  audio_utils.py
  features.py
  model.py              # CNN + BiLSTM
  dataset.py            # MELD loader
  manifest_dataset.py   # RAVDESS loader
  temporal_tracking.py  # novelty: stable regions + transitions
scripts/
  download_meld.py
  prepare_meld.py
  prepare_ravdess.py
  train.py
  evaluate.py
  infer_timeline.py
  make_demo_audio.py
  finish_meld_pipeline.py
```

## Viva note

- Continuous SER here = dense window predictions over time + timeline tracking.
- Official evaluation uses utterance/clip labels from the dataset.
- Contribution = temporal tracking module that builds interpretable emotion timelines.
