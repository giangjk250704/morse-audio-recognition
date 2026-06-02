# Precompute spectrogram .npy files for synthetic_realistic_v1 on CPU VM,
# pack them into tar.gz shards, and upload to GCS.
#
# Output shards:
#   gs://dl-ptit/data/synthetic_realistic_v1/specs/synreal_v1_specs_shard_000.tar.gz
#   gs://dl-ptit/data/synthetic_realistic_v1/specs/synreal_v1_specs_shard_001.tar.gz


import os
import re
import io
import json
import time
import tarfile
import shutil
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import librosa
import soundfile as sf
from google.cloud import storage


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--project_id", default="bigdataptit2026")
    p.add_argument("--bucket_name", default="dl-ptit")

    p.add_argument("--gcs_audio", default="data/synthetic_realistic_v1/audio")
    p.add_argument("--gcs_annotations", default="data/annotations")
    p.add_argument("--labels_json", default="synthetic_realistic_v1_labels.json")

    p.add_argument("--gcs_out_specs", default="data/synthetic_realistic_v1/specs")
    p.add_argument("--manifest_name", default="synreal_v1_specs_manifest.json")

    p.add_argument("--work_dir", default="/tmp/synreal_specs_work")
    p.add_argument("--max_samples", type=int, default=50000)
    p.add_argument("--shard_size", type=int, default=5000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)

    # Feature config must match training script/model.
    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--hop_length", type=int, default=128)
    p.add_argument("--n_bins", type=int, default=21)
    p.add_argument("--target_freq", type=float, default=800.0)

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep_local", action="store_true")

    return p.parse_args()


args = parse_args()

PROJECT_ID      = args.project_id
BUCKET_NAME     = args.bucket_name

GCS_AUDIO       = args.gcs_audio.rstrip("/")
GCS_ANNOTATIONS = args.gcs_annotations.rstrip("/")
GCS_OUT_SPECS   = args.gcs_out_specs.rstrip("/")
LABELS_JSON     = args.labels_json

WORK_DIR        = Path(args.work_dir)
MAX_SAMPLES     = args.max_samples
SHARD_SIZE      = args.shard_size
WORKERS         = args.workers

TARGET_SR       = args.target_sr
N_FFT           = args.n_fft
HOP_LENGTH      = args.hop_length
N_BINS          = args.n_bins
TARGET_FREQ     = args.target_freq

random.seed(args.seed)
np.random.seed(args.seed)

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

_thread_local = threading.local()


def get_gcs_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = storage.Client(project=PROJECT_ID)
    return _thread_local.client


def get_bucket():
    return get_gcs_client().bucket(BUCKET_NAME)


def extract_spectrogram(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)

    center = int(np.argmin(np.abs(freqs - TARGET_FREQ)))
    half = N_BINS // 2

    lo = max(0, center - half)
    hi = lo + N_BINS

    if hi > S.shape[0]:
        hi = S.shape[0]
        lo = max(0, hi - N_BINS)

    S_narrow = S[lo:hi, :]

    S_log = np.log1p(S_narrow)
    mean = S_log.mean(axis=1, keepdims=True)
    std = S_log.std(axis=1, keepdims=True) + 1e-9
    S_norm = (S_log - mean) / std

    return np.clip(S_norm, -3.0, 3.0).T.astype(np.float32)


def npy_name_for_wav(filename: str) -> str:
    return filename.replace(".wav", ".npy")


def process_one(record, shard_dir: Path):
    fname = record["filename"]
    npy_name = npy_name_for_wav(fname)
    out_path = shard_dir / npy_name

    if out_path.exists():
        return True, fname, ""

    try:
        audio_bytes = get_bucket().blob(f"{GCS_AUDIO}/{fname}").download_as_bytes()
        y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        spec = extract_spectrogram(y, sr)
        np.save(out_path, spec)
        return True, fname, ""

    except Exception as e:
        return False, fname, str(e)


def upload_file(local_path: Path, gcs_path: str):
    bucket.blob(gcs_path).upload_from_filename(str(local_path))


def blob_exists(gcs_path: str):
    return bucket.blob(gcs_path).exists()


def make_tar_gz(source_dir: Path, tar_path: Path):
    with tarfile.open(tar_path, "w:gz") as tar:
        for p in sorted(source_dir.glob("*.npy")):
            tar.add(str(p), arcname=p.name)


def human_size(num_bytes):
    n = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def run():
    t0 = time.time()

    print("=" * 90)
    print("PRECOMPUTE SYNREAL SPECS TO TAR SHARDS")
    print("=" * 90)
    print("PROJECT_ID:", PROJECT_ID)
    print("BUCKET_NAME:", BUCKET_NAME)
    print("GCS_AUDIO:", GCS_AUDIO)
    print("LABELS:", f"{GCS_ANNOTATIONS}/{LABELS_JSON}")
    print("GCS_OUT_SPECS:", GCS_OUT_SPECS)
    print("WORK_DIR:", WORK_DIR)
    print("MAX_SAMPLES:", MAX_SAMPLES)
    print("SHARD_SIZE:", SHARD_SIZE)
    print("WORKERS:", WORKERS)
    print("FEATURE:", dict(target_sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bins=N_BINS, target_freq=TARGET_FREQ))

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading labels...")
    records = json.loads(bucket.blob(f"{GCS_ANNOTATIONS}/{LABELS_JSON}").download_as_text())
    print("Total labels:", len(records))

    # Do not shuffle. Precompute all first MAX_SAMPLES so any random subset can still use specs
    # if max_samples <= available and full 50k is usually desired.
    records = records[:min(MAX_SAMPLES, len(records))]
    print("Using records:", len(records))

    shards = []
    for shard_idx, start in enumerate(range(0, len(records), SHARD_SIZE)):
        end = min(start + SHARD_SIZE, len(records))
        shards.append((shard_idx, start, end, records[start:end]))

    print("Total shards:", len(shards))

    manifest = {
        "labels_json": f"{GCS_ANNOTATIONS}/{LABELS_JSON}",
        "gcs_audio": GCS_AUDIO,
        "gcs_out_specs": GCS_OUT_SPECS,
        "max_samples": MAX_SAMPLES,
        "shard_size": SHARD_SIZE,
        "feature_config": {
            "target_sr": TARGET_SR,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_bins": N_BINS,
            "target_freq": TARGET_FREQ,
        },
        "shards": [],
    }

    total_ok = 0
    total_fail = 0

    for shard_idx, start, end, shard_records in shards:
        shard_name = f"synreal_v1_specs_shard_{shard_idx:03d}.tar.gz"
        gcs_tar_path = f"{GCS_OUT_SPECS}/{shard_name}"

        print("\n" + "-" * 90)
        print(f"[Shard {shard_idx:03d}] records {start}-{end-1} | count={len(shard_records)}")
        print("GCS:", gcs_tar_path)

        if blob_exists(gcs_tar_path) and not args.overwrite:
            print("Shard exists, skipping. Use --overwrite to rebuild.")
            manifest["shards"].append({
                "shard_index": shard_idx,
                "start": start,
                "end": end,
                "count": len(shard_records),
                "tar": gcs_tar_path,
                "status": "skipped_exists",
            })
            continue

        shard_dir = WORK_DIR / f"shard_{shard_idx:03d}"
        tar_path = WORK_DIR / shard_name

        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)

        if tar_path.exists():
            tar_path.unlink()

        ok_count = 0
        fail_count = 0
        failed_examples = []
        shard_t0 = time.time()

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = [ex.submit(process_one, r, shard_dir) for r in shard_records]

            for i, fut in enumerate(as_completed(futures), 1):
                ok, fname, err = fut.result()

                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                    if len(failed_examples) < 10:
                        failed_examples.append({"filename": fname, "error": err})

                if i % 500 == 0 or i == len(shard_records):
                    print(f"  {i}/{len(shard_records)} done | ok={ok_count} fail={fail_count}")

        print("Packing tar.gz...")
        make_tar_gz(shard_dir, tar_path)

        tar_size = tar_path.stat().st_size
        print("Tar size:", human_size(tar_size))

        print("Uploading shard...")
        upload_file(tar_path, gcs_tar_path)

        elapsed = time.time() - shard_t0
        print(f"Shard done in {elapsed:.1f}s | ok={ok_count} fail={fail_count}")

        total_ok += ok_count
        total_fail += fail_count

        manifest["shards"].append({
            "shard_index": shard_idx,
            "start": start,
            "end": end,
            "count": len(shard_records),
            "ok": ok_count,
            "fail": fail_count,
            "failed_examples": failed_examples,
            "tar": gcs_tar_path,
            "tar_size_bytes": tar_size,
            "elapsed_sec": round(elapsed, 3),
            "status": "done",
        })

        if not args.keep_local:
            shutil.rmtree(shard_dir, ignore_errors=True)
            tar_path.unlink(missing_ok=True)

        # Upload manifest checkpoint after each shard.
        manifest["total_ok_so_far"] = total_ok
        manifest["total_fail_so_far"] = total_fail
        manifest["elapsed_sec_so_far"] = round(time.time() - t0, 3)

        bucket.blob(f"{GCS_OUT_SPECS}/{args.manifest_name}").upload_from_string(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            content_type="application/json",
        )

    manifest["total_ok"] = total_ok
    manifest["total_fail"] = total_fail
    manifest["elapsed_sec"] = round(time.time() - t0, 3)

    bucket.blob(f"{GCS_OUT_SPECS}/{args.manifest_name}").upload_from_string(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        content_type="application/json",
    )

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print("Total ok:", total_ok)
    print("Total fail:", total_fail)
    print("Manifest:", f"gs://{BUCKET_NAME}/{GCS_OUT_SPECS}/{args.manifest_name}")
    print("Shards prefix:", f"gs://{BUCKET_NAME}/{GCS_OUT_SPECS}/")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run()
