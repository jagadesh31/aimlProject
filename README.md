# Continuous Speech Emotion Recognition

Audio-only emotion recognition on conversational speech, plus a timeline that shows when the emotion stays and when it changes.

The main dataset is **MELD** (*Friends* dialogues). **RAVDESS** was only an early pipeline test.

## Folder layout

```text
config.py                 settings (sample rate, emotions, paths)
src/                      model, features, dataset, timeline
scripts/                  commands you run
docs/submission/          review PDFs (introduction, diagram, algorithms)
checkpoints/
  best_model.pt           model used for the demo and the 49% test score
  best_model_meld.pt      same MELD weights
  previous/               earlier runs, including the 34% model
outputs/
  metrics/                accuracy, F1, and the older score files
  timelines/              emotion timelines
  demo/                   demo_sequence.wav (RAVDESS), test_dia0_stitched.wav (MELD)
docs/submission/          the three review PDFs
data/
  meld/labels/            train, dev, test CSVs
  meld/audio/             wav clips
  meld/raw/               original download
  MELD_repo/              official MELD readme and code
  ravdess/                early acted-speech set
  cache/                  features for the current model
  cache/legacy_mel/       older feature files, not used by the current model
```

## What the system does

1. Turn each utterance into a log-mel spectrogram plus delta and delta-delta.
2. CNN + BiLSTM predicts one of 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise.
3. At inference, a 2-second window slides every 1 second.
4. Weak flips are smoothed, then stable regions and short transitions are written as a timeline.

Training uses one MELD label per utterance. The timeline is the extra continuous output.

## Setup

```bash
pip install -r requirements.txt
```

## Commands

```bash
python scripts/train.py --source meld
python scripts/evaluate.py --source meld --split test
python scripts/infer_timeline.py --split test --dialogue-id 0
python scripts/make_demo_audio.py
python scripts/infer_timeline.py --audio outputs/demo/test_dia0_stitched.wav
python scripts/infer_timeline.py --audio outputs/demo/demo_sequence.wav
```

## MELD test results

| Run | Accuracy | Weighted F1 | Where |
|-----|----------|-------------|--------|
| First model | 0.34 | 0.29 | `outputs/metrics/test_metrics_previous.json` |
| Current model | **0.49** | **0.36** | `outputs/metrics/test_metrics.json` |
| Weighted loss, best epoch by dev macro F1 | 0.48 | **0.41** (macro F1 0.20, was 0.14) | `outputs/metrics/test_metrics_wce.json` |
| + dialogue context | 0.505 | 0.41 (macro F1 0.20) | `outputs/metrics/test_metrics_context_wce_ens.json` |
| **+ neutral bias tuned on dev (best)** | **0.507** | **0.41** (macro F1 0.19) | `outputs/metrics/test_metrics_context_wce_ens_tuned.json` |
| + dialogue context, past turns only | 0.499 | 0.40 | `outputs/metrics/test_metrics_context_wce_ens_causal.json` |

Dialogue context: the CNN-BiLSTM is frozen, and a small GRU reads its per-utterance
features across the whole MELD scene (plus a same-speaker-as-previous flag from the CSV).
Five GRUs with different seeds are averaged. Every choice was made on dev accuracy.

```bash
python scripts/train_dialogue_context.py --checkpoint checkpoints/best_model_meld_wce.pt --n-models 5 --tag context_wce_ens
python scripts/train_dialogue_context.py --checkpoint checkpoints/best_model_meld_wce.pt --n-models 5 --tune-neutral --tag context_wce_ens_tuned
```

Averaging four CNN-BiLSTM copies trained with different seeds (`best_model_meld_wce_s1..s3.pt`)
scored lower on dev, so it is not used.

The weighted-loss model is trained with:

```bash
python scripts/train.py --source meld --sampler none --loss wce --select macro_f1 --tag wce --epochs 20
python scripts/evaluate.py --source meld --split test --checkpoint checkpoints/best_model_meld_wce.pt --tag wce
```

Always guessing neutral scores about **0.48**, because neutral is almost half of MELD test. The current model is just above that line. Neutral recall is high. Anger is partly recognized. Joy, fear, disgust, and sadness are still weak. Say that directly if asked.

What changed:

- Features are log-mel plus delta and delta-delta, not a single spectrogram.
- Clips are no longer padded out to 6 seconds of silence.
- Training follows the real MELD class mix, with light label smoothing.
- The checkpoint used for demo and test is `checkpoints/best_model.pt`.
- An earlier balanced-sampling run is kept at `checkpoints/best_model_meld_balanced.pt`. It did not raise accuracy.
