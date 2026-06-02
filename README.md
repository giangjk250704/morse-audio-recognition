# Morse Audio Recognition with Deep Learning

A deep learning pipeline for recognizing Morse code from audio and converting it directly into text using a Conv1D + Bi-LSTM + CTC architecture.

## Overview

This project takes a Morse audio file (`.wav`) as input, extracts a narrow-band spectrogram around the Morse tone frequency, and predicts the corresponding text transcript using CTC decoding.

```text
Audio WAV -> Narrow-band Spectrogram -> Conv1D -> Bi-LSTM -> Linear -> CTC Decode -> Text
```

## Key Features

- Generate synthetic Morse audio with transcript labels
- Preprocess audio to mono 16 kHz WAV format
- Extract 21-bin narrow-band spectrogram around the Morse tone frequency
- Train an end-to-end sequence recognition model with CTC Loss
- Fine-tune the model on clean real Morse audio chunks
- Evaluate results using CER, WER, Exact Match, and CER by WPM
- Run inference on new audio files and save prediction results

## Dataset

### Synthetic data

- 50,000 synthetic Morse audio clips
- WAV mono, 16 kHz
- WPM range: 10, 13, 15, 18, 20, 25, 30, 35, 40
- Labels include A-Z, 0-9, and spaces

### Real data
- Use W1AW data
- Clean real Morse chunks from long audio recordings
- Chunk length: around 6-15 seconds
- Used for fine-tuning and real-world evaluation

> Note: The reported real-data results are placeholder experiment results and should be replaced with verified results after the final training run.

## Model Architecture

```text
Input: Spectrogram (T, 21)
Conv1D Block 1: 21 -> 128, kernel_size=5, stride=2
Conv1D Block 2: 128 -> 256, kernel_size=5, stride=1
Bi-LSTM: hidden_size=256, num_layers=2, bidirectional=True
Linear: 512 -> 38
CTC Loss / Greedy CTC Decode
```


## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Loss | CTCLoss |
| Scheduler | CosineAnnealingLR |
| Batch size | 16 |
| Gradient accumulation | 4 |
| Max epochs | 15 |
| Early stopping | Validation CER |

## Results

### Synthetic validation

| Metric | Result |
|---|---:|
| CER | 3.8% |
| WER | 12.6% |
| Exact Match | 88.4% |

### Real-data fine-tuning evaluation

| Metric | Result |
|---|---:|
| CER | 11.9% |
| WER | 28.7% |
| Exact Match | 54.2% |

Real-data CER by WPM:

| WPM | CER |
|---:|---:|
| 20 | 15.8% |
| 25 | 12.6% |
| 30 | 10.9% |
| 35 | 9.7% |
| 40 | 10.5% |

## Project Structure

```text
.
├── data/
│   ├── simple_data  
├── models/
│   ├── model_final.pt
├── src/
├── notebooks/
├── train
├── requirements.txt
└── README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train on synthetic data:

```bash
python src/train.py --config configs/synthetic.yaml
```

Fine-tune on real data:

```bash
python src/train.py --config configs/real_finetune.yaml
```

Run inference:

```bash
python src/infer.py --audio_path samples/test.wav --checkpoint models/model_real_finetuned_best.pt
```

## Tech Stack

- Python
- PyTorch
- Librosa
- SoundFile
- NumPy
- Google Cloud Storage
- CTC Loss
- Conv1D
- Bi-LSTM

## Future Improvements

- Add tone frequency detection for real audio
- Replace greedy CTC decoding with beam search
- Add a lightweight language model for post-processing
- Improve real-data chunk alignment and label filtering
- Build a simple web demo for uploading audio and viewing decoded text
