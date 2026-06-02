# Goal:
#   Create synthetic Morse with exact labels, but audio distribution closer to W1AW:
#   - WPM diversity
#   - timing jitter
#   - tone frequency jitter around 800 Hz
#   - symbol gain variation
#   - attack/release envelope
#   - random pauses
#   - noise / hum / clipping / small DC offset
#
# Input:
#   gs://dl-ptit/data/annotations/audio_labels_final.json
#   gs://dl-ptit/data/Labels ARRL/*.txt
#
# Output:
#   gs://dl-ptit/data/synthetic_realistic_v1/audio/*.wav
#   gs://dl-ptit/data/annotations/synthetic_realistic_v1_labels.json
#   gs://dl-ptit/data/annotations/synthetic_realistic_v1_stats.json


import os
import re
import io
import json
import math
import time
import random
import argparse
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
from google.cloud import storage


# ============================================================
# ARGS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--project_id", default="bigdataptit2026")
    p.add_argument("--bucket_name", default="dl-ptit")

    p.add_argument("--gcs_labels", default="data/Labels ARRL")
    p.add_argument("--gcs_annotations", default="data/annotations")
    p.add_argument("--source_json", default="audio_labels_final.json")

    p.add_argument("--out_audio", default="data/synthetic_realistic_v1/audio")
    p.add_argument("--out_labels", default="synthetic_realistic_v1_labels.json")
    p.add_argument("--out_stats", default="synthetic_realistic_v1_stats.json")
    p.add_argument("--out_corpus", default="synthetic_realistic_v1_corpus.json")

    p.add_argument("--target_samples", type=int, default=2000)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--min_duration", type=float, default=5.0)
    p.add_argument("--max_duration", type=float, default=15.5)
    p.add_argument("--target_duration_min", type=float, default=7.0)
    p.add_argument("--target_duration_max", type=float, default=13.0)

    p.add_argument("--min_chars", type=int, default=10)
    p.add_argument("--max_chars", type=int, default=120)

    # Use WPM weights. This default emphasizes 18-40, but still includes slow WPM.
    p.add_argument(
        "--wpm_weights",
        default="5:0.04,10:0.06,13:0.06,15:0.08,18:0.12,20:0.12,25:0.14,30:0.14,35:0.14,40:0.10",
    )

    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


args = parse_args()


# ============================================================
# CONFIG
# ============================================================

PROJECT_ID      = args.project_id
BUCKET_NAME     = args.bucket_name

GCS_LABELS      = args.gcs_labels.rstrip("/")
GCS_ANNOTATIONS = args.gcs_annotations.rstrip("/")
SOURCE_JSON     = args.source_json

OUT_AUDIO       = args.out_audio.rstrip("/")
OUT_LABELS      = args.out_labels
OUT_STATS       = args.out_stats
OUT_CORPUS      = args.out_corpus

TARGET_SAMPLES  = args.target_samples
SR              = args.sr
WORKERS         = args.workers
BATCH_SIZE      = args.batch_size

MIN_DURATION    = args.min_duration
MAX_DURATION    = args.max_duration
TARGET_DUR_MIN  = args.target_duration_min
TARGET_DUR_MAX  = args.target_duration_max

MIN_CHARS       = args.min_chars
MAX_CHARS       = args.max_chars

random.seed(args.seed)
np.random.seed(args.seed)


# ============================================================
# GCS
# ============================================================

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)


# ============================================================
# MORSE TABLE
# ============================================================

MORSE_TABLE = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
}

VALID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_label(raw: str):
    if raw[:3] == "GIF" or "\x00" in raw[:100]:
        return None, None

    wpm = None
    m = re.search(r"NOW\s+(\d+)\s+WPM", raw, re.IGNORECASE)
    if m:
        wpm = int(m.group(1))

    normalized = re.sub(r"[^\w\s]", "=", raw)

    text = re.sub(
        r"=\s*NOW\s+\d+\s+WPM\s*=\s*TEXT\s+IS\s+FROM[\s\S]*?PAGE\s+\d+\s*=\s*",
        "",
        normalized,
        count=1,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"=\s*NOW\s+\d+\s+WPM\s+TRANSITION\s+FILE\s+FOLLOWS\s*=\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"=\s*NOW[\s\S]*?=\s*", "", text, count=1)
    text = re.sub(r"=\s*END\s+OF.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"QST\s+DE\s+W1AW.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)

    text = re.sub(r"[=<>_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().upper()

    return text, wpm


def filter_transcription(text: str) -> str:
    filtered = "".join(c for c in text.upper() if c in VALID_CHARS)
    return re.sub(r" +", " ", filtered).strip()


def parse_wpm_weights(s):
    pairs = []
    for part in s.split(","):
        k, v = part.split(":")
        pairs.append((int(k.strip()), float(v.strip())))

    wpms = [x[0] for x in pairs]
    weights = np.array([x[1] for x in pairs], dtype=np.float64)
    weights = weights / weights.sum()
    return wpms, weights


WPM_VALUES, WPM_WEIGHTS = parse_wpm_weights(args.wpm_weights)


# ============================================================
# CORPUS
# ============================================================

def load_or_build_corpus():
    corpus_blob = bucket.blob(f"{GCS_ANNOTATIONS}/{OUT_CORPUS}")

    if corpus_blob.exists() and not args.overwrite:
        print("[Corpus] Loading existing corpus:", f"{GCS_ANNOTATIONS}/{OUT_CORPUS}")
        return json.loads(corpus_blob.download_as_text())

    print("[Corpus] Building corpus from ARRL labels...")

    all_records = json.loads(
        bucket.blob(f"{GCS_ANNOTATIONS}/{SOURCE_JSON}").download_as_text()
    )

    docs = []
    bad = 0

    for i, rec in enumerate(all_records, 1):
        fname = rec["filename"]
        label_fname = fname.replace(".wav", ".txt")

        try:
            raw_bytes = bucket.blob(f"{GCS_LABELS}/{label_fname}").download_as_bytes()
            raw_text = raw_bytes.decode("utf-8", errors="replace")

            text, wpm = clean_label(raw_text)
            if text is None:
                bad += 1
                continue

            text = filter_transcription(text)
            words = [w for w in text.split() if any(c in MORSE_TABLE for c in w)]

            if len(words) < 5:
                bad += 1
                continue

            docs.append({
                "source_file": fname,
                "wpm_label": wpm,
                "words": words,
                "word_count": len(words),
            })

        except Exception:
            bad += 1

        if i % 500 == 0:
            print(f"  processed {i}/{len(all_records)} labels | docs={len(docs)} bad={bad}")

    corpus = {
        "docs": docs,
        "created_from": f"{GCS_ANNOTATIONS}/{SOURCE_JSON}",
        "doc_count": len(docs),
        "bad_count": bad,
    }

    corpus_blob.upload_from_string(
        json.dumps(corpus, ensure_ascii=False, indent=2),
        content_type="application/json",
    )

    print("[Corpus] Uploaded:", f"{GCS_ANNOTATIONS}/{OUT_CORPUS}")
    print("[Corpus] docs:", len(docs), "bad:", bad)

    return corpus


# ============================================================
# MORSE SYNTHESIS
# ============================================================

def farnsworth_params(wpm: int):
    """
    Returns dot_sec, intra_gap, char_gap, word_gap.
    For <=15 WPM use Farnsworth-like char speed with randomization later.
    """
    if wpm <= 15:
        char_wpm = 15
        dot = 1.2 / char_wpm
        char_time = 31 * dot
        extra = (60.0 / wpm - char_time) / 19
        char_gap = 3 * dot + 3 * extra
        word_gap = 7 * dot + 7 * extra
    else:
        dot = 1.2 / wpm
        char_gap = 3 * dot
        word_gap = 7 * dot

    intra_gap = dot
    return dot, intra_gap, char_gap, word_gap


def apply_attack_release(tone, sr, attack_ms, release_ms):
    n = len(tone)
    if n <= 2:
        return tone

    attack_n = min(n // 2, max(1, int(sr * attack_ms / 1000.0)))
    release_n = min(n // 2, max(1, int(sr * release_ms / 1000.0)))

    env = np.ones(n, dtype=np.float32)

    if attack_n > 0:
        env[:attack_n] = np.linspace(0.0, 1.0, attack_n, dtype=np.float32)
    if release_n > 0:
        env[-release_n:] = np.linspace(1.0, 0.0, release_n, dtype=np.float32)

    return tone * env


def synth_tone(duration, sr, freq, phase, amp, attack_ms, release_ms):
    n = max(1, int(round(duration * sr)))
    t = np.arange(n, dtype=np.float32) / sr
    tone = amp * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
    tone = apply_attack_release(tone, sr, attack_ms, release_ms)

    # Continue phase
    phase = (phase + 2 * np.pi * freq * (n / sr)) % (2 * np.pi)
    return tone, phase


def silence(duration, sr):
    n = max(0, int(round(duration * sr)))
    return np.zeros(n, dtype=np.float32)


def add_noise_and_effects(y, sr, rng):
    y = y.astype(np.float32)

    # Random gain
    y *= rng.uniform(0.35, 0.95)

    # Add low-frequency hum sometimes.
    if rng.random() < 0.35:
        hum_freq = rng.choice([50.0, 60.0, 100.0, 120.0])
        t = np.arange(len(y), dtype=np.float32) / sr
        hum_amp = rng.uniform(0.002, 0.018)
        y += hum_amp * np.sin(2 * np.pi * hum_freq * t).astype(np.float32)

    # Add white noise by SNR.
    snr_db = rng.uniform(5.0, 25.0)
    sig_power = float(np.mean(y * y)) + 1e-9
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0.0, math.sqrt(noise_power), size=len(y)).astype(np.float32)
    y = y + noise

    # Occasional background hiss independent of signal.
    if rng.random() < 0.40:
        hiss = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0.0, rng.uniform(0.001, 0.006), size=len(y)).astype(np.float32)
        y += hiss

    # Slow amplitude fading.
    if rng.random() < 0.55 and len(y) > sr:
        fade_freq = rng.uniform(0.05, 0.35)
        depth = rng.uniform(0.05, 0.25)
        t = np.arange(len(y), dtype=np.float32) / sr
        fade = 1.0 + depth * np.sin(2 * np.pi * fade_freq * t + rng.uniform(0, 2*np.pi))
        y *= fade.astype(np.float32)

    # Small DC offset sometimes.
    if rng.random() < 0.25:
        y += rng.uniform(-0.01, 0.01)

    # Mild clipping sometimes.
    if rng.random() < 0.18:
        clip_val = rng.uniform(0.55, 0.95)
        y = np.clip(y, -clip_val, clip_val)

    # Normalize safe.
    peak = np.max(np.abs(y)) + 1e-9
    if peak > 0.98:
        y = y / peak * rng.uniform(0.80, 0.95)

    return y.astype(np.float32), snr_db


def synthesize_morse_text(text, wpm, sr, rng):
    dot, intra_gap, char_gap, word_gap = farnsworth_params(wpm)

    # Per-sample base perturbation.
    dot *= rng.uniform(0.94, 1.06)
    intra_gap *= rng.uniform(0.90, 1.10)
    char_gap *= rng.uniform(0.85, 1.25)
    word_gap *= rng.uniform(0.80, 1.45)

    freq = rng.uniform(780.0, 835.0)
    attack_ms = rng.uniform(3.0, 14.0)
    release_ms = rng.uniform(4.0, 20.0)

    phase = rng.uniform(0.0, 2 * np.pi)
    chunks = []

    # Leading silence / pre-roll
    chunks.append(silence(rng.uniform(0.02, 0.35), sr))

    words = text.split()

    for wi, word in enumerate(words):
        valid_chars = [c for c in word if c in MORSE_TABLE]

        for ci, c in enumerate(valid_chars):
            morse = MORSE_TABLE[c]

            for si, sym in enumerate(morse):
                if sym == ".":
                    dur = dot * rng.uniform(0.82, 1.18)
                else:
                    dur = 3.0 * dot * rng.uniform(0.82, 1.18)

                amp = rng.uniform(0.75, 1.25)
                tone, phase = synth_tone(dur, sr, freq * rng.uniform(0.997, 1.003), phase, amp, attack_ms, release_ms)
                chunks.append(tone)

                if si < len(morse) - 1:
                    gap = intra_gap * rng.uniform(0.75, 1.35)
                    chunks.append(silence(gap, sr))

            if ci < len(valid_chars) - 1:
                gap = char_gap * rng.uniform(0.70, 1.45)
                chunks.append(silence(gap, sr))

        if wi < len(words) - 1:
            gap = word_gap * rng.uniform(0.65, 1.75)

            # Occasional phrase pause.
            if rng.random() < 0.08:
                gap += rng.uniform(0.15, 1.25)

            chunks.append(silence(gap, sr))

    # Trailing silence
    chunks.append(silence(rng.uniform(0.02, 0.35), sr))

    y = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)

    y, snr_db = add_noise_and_effects(y, sr, rng)

    meta = {
        "wpm": int(wpm),
        "tone_freq": round(float(freq), 3),
        "dot_sec_base": round(float(dot), 6),
        "attack_ms": round(float(attack_ms), 3),
        "release_ms": round(float(release_ms), 3),
        "snr_db": round(float(snr_db), 3),
    }

    return y, meta


def estimate_text_duration(words, wpm):
    # Rough duration estimate to pick text window.
    dot, intra_gap, char_gap, word_gap = farnsworth_params(wpm)

    dur = 0.15
    for wi, word in enumerate(words):
        chars = [c for c in word if c in MORSE_TABLE]
        for ci, c in enumerate(chars):
            morse = MORSE_TABLE[c]
            for si, sym in enumerate(morse):
                dur += dot if sym == "." else 3 * dot
                if si < len(morse) - 1:
                    dur += intra_gap
            if ci < len(chars) - 1:
                dur += char_gap
        if wi < len(words) - 1:
            dur += word_gap
    dur += 0.15
    return dur


def choose_text_window(corpus_docs, wpm, rng):
    # Try multiple times to find a duration-compatible text segment.
    for _ in range(80):
        doc = rng.choice(corpus_docs)
        words = doc["words"]

        if len(words) < 4:
            continue

        start = rng.randint(0, max(0, len(words) - 2))

        target_dur = rng.uniform(TARGET_DUR_MIN, TARGET_DUR_MAX)

        current = []
        for j in range(start, min(len(words), start + 80)):
            current.append(words[j])
            est_dur = estimate_text_duration(current, wpm)

            if est_dur >= target_dur:
                text = " ".join(current)
                text = filter_transcription(text)
                if MIN_CHARS <= len(text) <= MAX_CHARS:
                    return text, doc["source_file"], est_dur
                break

    # Fallback: short random phrase.
    doc = rng.choice(corpus_docs)
    words = doc["words"]
    start = rng.randint(0, max(0, len(words) - 4))
    n = rng.randint(3, min(12, len(words) - start))
    text = filter_transcription(" ".join(words[start:start+n]))
    return text, doc["source_file"], estimate_text_duration(text.split(), wpm)


# ============================================================
# SAMPLE GENERATION
# ============================================================

def generate_one(index, corpus_docs):
    rng = random.Random(args.seed + index * 7919)

    # Try hard to get a valid sample. If one WPM/text window is too long,
    # resample another one instead of accepting a bad duration.
    last_error = None

    for attempt in range(120):
        try:
            wpm = int(rng.choices(WPM_VALUES, weights=WPM_WEIGHTS, k=1)[0])
            text, source_file, est_dur = choose_text_window(corpus_docs, wpm, rng)

            if not text or not (MIN_CHARS <= len(text) <= MAX_CHARS):
                continue

            y, meta = synthesize_morse_text(text, wpm, SR, rng)
            duration = len(y) / SR

            if not (MIN_DURATION <= duration <= MAX_DURATION):
                continue

            fname = f"synreal_v1_{index:06d}_{wpm}wpm.wav"

            wav_io = io.BytesIO()
            sf.write(wav_io, y, SR, format="WAV", subtype="FLOAT")
            wav_bytes = wav_io.getvalue()

            record = {
                "filename": fname,
                "transcription": text,
                "metadata": {
                    "audio_type": "synthetic_realistic",
                    "source": "ARRL_text_synthetic",
                    "source_text_file": source_file,
                    "wpm": wpm,
                    "duration_sec": round(float(duration), 4),
                    "sample_rate": SR,
                    "label_method": "synthetic_exact_realistic_v1",
                    **meta,
                },
            }

            return fname, wav_bytes, record

        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Could not generate valid sample after 120 attempts. Last error: {last_error}")


# ============================================================
# RESUME / OUTPUT
# ============================================================

def load_existing_records():
    labels_blob = bucket.blob(f"{GCS_ANNOTATIONS}/{OUT_LABELS}")

    if args.overwrite:
        return []

    if args.resume and labels_blob.exists():
        print("[Resume] Loading existing labels:", f"{GCS_ANNOTATIONS}/{OUT_LABELS}")
        return json.loads(labels_blob.download_as_text())

    return []


def upload_checkpoint(records, stats):
    bucket.blob(f"{GCS_ANNOTATIONS}/{OUT_LABELS}").upload_from_string(
        json.dumps(records, ensure_ascii=False, indent=2),
        content_type="application/json",
    )
    bucket.blob(f"{GCS_ANNOTATIONS}/{OUT_STATS}").upload_from_string(
        json.dumps(stats, ensure_ascii=False, indent=2),
        content_type="application/json",
    )


# ============================================================
# MAIN
# ============================================================

def run():
    t0 = time.time()

    print("=" * 90)
    print("SYNTHETIC REALISTIC V1 GENERATOR")
    print("=" * 90)
    print("PROJECT_ID:", PROJECT_ID)
    print("BUCKET_NAME:", BUCKET_NAME)
    print("TARGET_SAMPLES:", TARGET_SAMPLES)
    print("OUT_AUDIO:", OUT_AUDIO)
    print("OUT_LABELS:", f"{GCS_ANNOTATIONS}/{OUT_LABELS}")
    print("SR:", SR)
    print("WORKERS:", WORKERS)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("WPM_VALUES:", WPM_VALUES)
    print("WPM_WEIGHTS:", WPM_WEIGHTS.tolist())

    corpus = load_or_build_corpus()
    corpus_docs = corpus["docs"]

    if not corpus_docs:
        raise RuntimeError("Empty corpus")

    records = load_existing_records()
    existing = len(records)

    print("Existing records:", existing)

    start_idx = existing
    target_end = TARGET_SAMPLES

    if start_idx >= target_end:
        print("Already done.")
        return

    generated = 0
    failed = 0

    idx = start_idx

    while idx < target_end:
        batch_indices = list(range(idx, min(idx + BATCH_SIZE, target_end)))
        idx = batch_indices[-1] + 1

        print(f"\n[Batch] {batch_indices[0]} - {batch_indices[-1]}")

        batch_t0 = time.time()
        batch_records = []

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = [ex.submit(generate_one, i, corpus_docs) for i in batch_indices]

            for fut in as_completed(futures):
                try:
                    fname, wav_bytes, rec = fut.result()

                    bucket.blob(f"{OUT_AUDIO}/{fname}").upload_from_string(
                        wav_bytes,
                        content_type="audio/wav",
                    )

                    batch_records.append(rec)
                    generated += 1

                except Exception as e:
                    failed += 1
                    if failed <= 10:
                        print("FAILED:", e)

        # Keep deterministic-ish order by filename.
        batch_records.sort(key=lambda r: r["filename"])
        records.extend(batch_records)

        by_wpm = defaultdict(int)
        durations = []

        for r in records:
            by_wpm[str(r["metadata"]["wpm"])] += 1
            durations.append(float(r["metadata"]["duration_sec"]))

        stats = {
            "target_samples": TARGET_SAMPLES,
            "current_samples": len(records),
            "generated_this_run": generated,
            "failed_this_run": failed,
            "chunks_by_wpm": dict(by_wpm),
            "duration_mean": float(np.mean(durations)) if durations else None,
            "duration_median": float(np.median(durations)) if durations else None,
            "duration_min": float(np.min(durations)) if durations else None,
            "duration_max": float(np.max(durations)) if durations else None,
            "sr": SR,
            "out_audio": OUT_AUDIO,
            "out_labels": f"{GCS_ANNOTATIONS}/{OUT_LABELS}",
            "elapsed_sec": round(time.time() - t0, 3),
        }

        upload_checkpoint(records, stats)

        dur_med = stats["duration_median"]
        dur_med_str = f"{dur_med:.2f}s" if dur_med is not None else "None"

        print(
            f"Batch done in {time.time() - batch_t0:.1f}s | "
            f"total={len(records)}/{TARGET_SAMPLES} | "
            f"generated={generated} failed={failed} | "
            f"duration_median={dur_med_str}"
        )

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print("Final samples:", len(records))
    print("Audio:", f"gs://{BUCKET_NAME}/{OUT_AUDIO}/")
    print("Labels:", f"gs://{BUCKET_NAME}/{GCS_ANNOTATIONS}/{OUT_LABELS}")
    print("Stats:", f"gs://{BUCKET_NAME}/{GCS_ANNOTATIONS}/{OUT_STATS}")


if __name__ == "__main__":
    run()
