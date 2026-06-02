# ============================================================
# W1AW Scaled Chunker V1
# ============================================================
# Re-create higher-quality W1AW real chunks using:
#   1) theoretical Morse word timing
#   2) per-file timing scale to actual audio duration
#   3) short chunks, default 12s
#   4) optional boundary snapping to low-RMS gaps
#
# Example debug run:
#   python build_scaled_chunks_v1.py --target_chunks 1000 --wpms 25,30,35,40 --overwrite
#
# Example expanded run:
#   python build_scaled_chunks_v1.py --target_chunks 5000 --wpms 18,20,25,30,35,40 --overwrite
# ============================================================

import os
import re
import json
import time
import random
import argparse
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
from google.cloud import storage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_id", default="bigdataptit2026")
    p.add_argument("--bucket_name", default="dl-ptit")
    p.add_argument("--gcs_processed", default="data/processed/audio")
    p.add_argument("--gcs_labels", default="data/Labels ARRL")
    p.add_argument("--gcs_annotations", default="data/annotations")
    p.add_argument("--gcs_out_audio", default="data/splits/audio_scaled_v1")
    p.add_argument("--source_json", default="audio_labels_final.json")
    p.add_argument("--out_labels", default="symbol_chunked_scaled_v1_labels.json")
    p.add_argument("--out_report", default="symbol_chunked_scaled_v1_file_report.jsonl")
    p.add_argument("--out_stats", default="symbol_chunked_scaled_v1_stats.json")
    p.add_argument("--wpms", default="25,30,35,40")
    p.add_argument("--target_chunks", type=int, default=1000)
    p.add_argument("--max_files_per_wpm", type=int, default=40)
    p.add_argument("--chunk_sec", type=float, default=12.0)
    p.add_argument("--min_chunk_sec", type=float, default=4.0)
    p.add_argument("--max_chunk_sec", type=float, default=15.0)
    p.add_argument("--snap_window_sec", type=float, default=0.80)
    p.add_argument("--snap_frame_ms", type=int, default=20)
    p.add_argument("--scale_min", type=float, default=0.85)
    p.add_argument("--scale_max", type=float, default=1.30)
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--batch_files", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_snap", action="store_true")
    return p.parse_args()


args = parse_args()
PROJECT_ID = args.project_id
BUCKET_NAME = args.bucket_name
GCS_PROCESSED = args.gcs_processed.rstrip("/")
GCS_LABELS = args.gcs_labels.rstrip("/")
GCS_ANNOTATIONS = args.gcs_annotations.rstrip("/")
GCS_OUT_AUDIO = args.gcs_out_audio.rstrip("/")
SOURCE_JSON = args.source_json
OUT_LABELS = args.out_labels
OUT_REPORT = args.out_report
OUT_STATS = args.out_stats
USE_WPMS = [int(x.strip()) for x in args.wpms.split(",") if x.strip()]
TARGET_CHUNKS = args.target_chunks
MAX_FILES_PER_WPM = args.max_files_per_wpm
CHUNK_SEC = args.chunk_sec
MIN_CHUNK_SEC = args.min_chunk_sec
MAX_CHUNK_SEC = args.max_chunk_sec
SNAP_WINDOW_SEC = args.snap_window_sec
SNAP_FRAME_MS = args.snap_frame_ms
USE_SNAP = not args.no_snap
SCALE_MIN = args.scale_min
SCALE_MAX = args.scale_max
MAX_WORKERS = args.max_workers
BATCH_FILES = args.batch_files
SILENCE_THRESH = 0.01

random.seed(args.seed)
np.random.seed(args.seed)

print("=" * 90)
print("W1AW SCALED CHUNKER V1")
print("=" * 90)
print("Input audio:", GCS_PROCESSED)
print("Input labels:", GCS_LABELS)
print("Source JSON:", f"{GCS_ANNOTATIONS}/{SOURCE_JSON}")
print("Output audio:", GCS_OUT_AUDIO)
print("Output labels:", f"{GCS_ANNOTATIONS}/{OUT_LABELS}")
print("WPMs:", USE_WPMS)
print("Target chunks:", TARGET_CHUNKS)
print("Chunk sec:", CHUNK_SEC)
print("Snap:", USE_SNAP)
print("Scale min/max:", SCALE_MIN, SCALE_MAX)
print("Workers:", MAX_WORKERS)

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)
_thread_local = threading.local()


def get_gcs_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = storage.Client(project=PROJECT_ID)
    return _thread_local.client


def get_bucket():
    return get_gcs_client().bucket(BUCKET_NAME)


MORSE_TABLE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    'BT': '-...-', 'AR': '.-.-.', 'SK': '...-.-', 'KN': '-.--.',
}
HEADER_PHRASES = [
    {"NOW", "WPM"}, {"TEXT", "FROM", "PAGE"},
    {"TRANSITION", "FILE", "FOLLOWS"}, {"QST", "DE", "W1AW"},
]


def encode_word(word: str) -> str:
    upper = word.upper()
    if upper in MORSE_TABLE:
        return MORSE_TABLE[upper]
    return " ".join(MORSE_TABLE[c] for c in upper if c in MORSE_TABLE)


def get_timing_params(wpm: int):
    if wpm <= 15:
        char_unit = 1.2 / 15
        char_time = 31 * char_unit
        extra = (60.0 / wpm - char_time) / 19
        char_gap = 3 * char_unit + 3 * extra
        word_gap = 7 * char_unit + 7 * extra
    else:
        char_unit = 1.2 / wpm
        char_gap = 3 * char_unit
        word_gap = 7 * char_unit
    return char_unit, char_gap, word_gap


def duration_of_symbol_string(morse_str: str, dot_sec: float) -> float:
    dur = 0.0
    for i, sym in enumerate(morse_str):
        dur += dot_sec if sym == "." else 3 * dot_sec
        if i < len(morse_str) - 1:
            dur += dot_sec
    return dur


def calculate_word_timings(transcription: str, wpm: int, start_offset: float = 0.0):
    dot_sec, char_gap, word_gap = get_timing_params(wpm)
    words = transcription.upper().split()
    timings, cursor = [], start_offset
    for wi, word in enumerate(words):
        word_start = cursor
        morse_word = encode_word(word)
        if not morse_word:
            cursor += dot_sec
            timings.append({"word": word, "start_sec": word_start, "end_sec": cursor})
            if wi < len(words) - 1:
                cursor += word_gap
            continue
        for ci, char_morse in enumerate(morse_word.split(" ")):
            cursor += duration_of_symbol_string(char_morse, dot_sec)
            if ci < len(morse_word.split(" ")) - 1:
                cursor += char_gap
        timings.append({"word": word, "start_sec": word_start, "end_sec": cursor})
        if wi < len(words) - 1:
            cursor += word_gap
    return timings


def calculate_duration(text: str, wpm: int) -> float:
    timings = calculate_word_timings(text, wpm, 0.0)
    return timings[-1]["end_sec"] if timings else 0.0


def clean_label(raw: str):
    if raw[:3] == "GIF" or "\x00" in raw[:100]:
        return None, None
    wpm = None
    m = re.search(r"NOW\s+(\d+)\s+WPM", raw, re.IGNORECASE)
    if m:
        wpm = int(m.group(1))
    normalized = re.sub(r"[^\w\s]", "=", raw)
    text = re.sub(r"=\s*NOW\s+\d+\s+WPM\s*=\s*TEXT\s+IS\s+FROM[\s\S]*?PAGE\s+\d+\s*=\s*", "", normalized, count=1, flags=re.IGNORECASE)
    text = re.sub(r"=\s*NOW\s+\d+\s+WPM\s+TRANSITION\s+FILE\s+FOLLOWS\s*=\s*", "", text, count=1, flags=re.IGNORECASE)
    text = re.sub(r"=\s*NOW[\s\S]*?=\s*", "", text, count=1)
    text = re.sub(r"=\s*END\s+OF.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"QST\s+DE\s+W1AW.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"[=<>_]", "", text)
    text = re.sub(r"\s+", " ", text).strip().upper()
    return text, wpm


def filter_transcription(text: str) -> str:
    valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
    return re.sub(r" +", " ", "".join(c for c in text.upper() if c in valid)).strip()


def is_header_chunk(transcription: str) -> bool:
    words = set(transcription.upper().split())
    return any(phrase.issubset(words) for phrase in HEADER_PHRASES)


def calculate_header_offset(raw_text: str, wpm: int) -> float:
    normalized = re.sub(r"[^\w\s=]", "=", raw_text)
    m = re.search(r"=\s*NOW\s+\d+\s+WPM\s*=\s*TEXT\s+IS\s+FROM[\s\S]*?PAGE\s+\d+\s*=", normalized, re.IGNORECASE)
    if not m:
        m = re.search(r"=\s*NOW\s+\d+\s+WPM\s+TRANSITION\s+FILE\s+FOLLOWS\s*=", normalized, re.IGNORECASE)
    if not m:
        m = re.search(r"=\s*NOW[\s\S]*?=", normalized)
        if not m:
            return 0.0
    header_text = m.group(0)
    header_clean = re.sub(r"=", " BT ", header_text.upper())
    header_clean = re.sub(r"[^A-Z0-9 ]", " ", header_clean)
    header_clean = re.sub(r"\s+", " ", header_clean).strip()
    return calculate_duration(header_clean, wpm)


def detect_audio_onset(y: np.ndarray, sr: int, frame_ms: int = 100, threshold: float = SILENCE_THRESH) -> float:
    frame_len = max(1, int(frame_ms / 1000 * sr))
    for i in range(0, max(1, len(y) - frame_len), frame_len):
        frame = y[i:i + frame_len]
        if len(frame) and float(np.sqrt(np.mean(frame ** 2))) > threshold:
            return i / sr
    return 0.0


def get_wpm_from_filename(filename: str):
    m = re.search(r"(\d+)wpm", filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


def compute_rms_envelope(y: np.ndarray, sr: int, frame_ms: int = 20):
    frame_len = max(1, int(sr * frame_ms / 1000))
    if len(y) < frame_len:
        return np.array([0.0], dtype=np.float32), np.array([float(np.sqrt(np.mean(y ** 2)))], dtype=np.float32)
    n = 1 + (len(y) - frame_len) // frame_len
    rms = np.empty(n, dtype=np.float32)
    times = np.empty(n, dtype=np.float32)
    for idx in range(n):
        start = idx * frame_len
        frame = y[start:start + frame_len]
        rms[idx] = float(np.sqrt(np.mean(frame ** 2)))
        times[idx] = start / sr
    return times, rms


def snap_to_low_energy(t: float, times: np.ndarray, rms: np.ndarray, total_duration: float, window_sec: float, prefer: str):
    if len(times) == 0:
        return max(0.0, min(t, total_duration))
    if prefer == "start":
        lo, hi = max(0.0, t - window_sec), min(total_duration, t + window_sec * 0.5)
    else:
        lo, hi = max(0.0, t - window_sec * 0.5), min(total_duration, t + window_sec)
    idxs = np.where((times >= lo) & (times <= hi))[0]
    if len(idxs) == 0:
        return max(0.0, min(t, total_duration))
    return float(times[idxs[int(np.argmin(rms[idxs]))]])


def scale_word_timings(word_timings, offset: float, total_duration: float):
    if not word_timings:
        return [], None, None, None
    theory_content = word_timings[-1]["end_sec"] - offset
    actual_content = total_duration - offset
    if theory_content <= 0 or actual_content <= 0:
        return [], None, theory_content, actual_content
    scale = actual_content / theory_content
    scaled = []
    for w in word_timings:
        s = offset + (w["start_sec"] - offset) * scale
        e = offset + (w["end_sec"] - offset) * scale
        scaled.append({"word": w["word"], "start_sec": max(0.0, s), "end_sec": min(total_duration, e)})
    return scaled, scale, theory_content, actual_content


def build_chunks(y, sr, timings, source_filename, wpm, offset, scale, header_offset, pre_roll):
    total_duration = len(y) / sr
    if USE_SNAP:
        times, rms = compute_rms_envelope(y, sr, SNAP_FRAME_MS)
    else:
        times, rms = np.array([]), np.array([])
    chunks, word_idx, chunk_idx = [], 0, 0
    while word_idx < len(timings):
        chunk_start = timings[word_idx]["start_sec"]
        if chunk_start >= total_duration:
            break
        target_end = min(chunk_start + CHUNK_SEC, total_duration)
        chunk_words, last_widx, actual_end = [], word_idx, chunk_start
        for i in range(word_idx, len(timings)):
            w = timings[i]
            if w["end_sec"] <= target_end:
                chunk_words.append(w["word"])
                actual_end = w["end_sec"]
                last_widx = i
            else:
                break
        if not chunk_words:
            word_idx += 1
            continue
        transcription = " ".join(chunk_words)
        if is_header_chunk(transcription) or len(transcription.split()) < 2:
            word_idx = last_widx + 1
            continue
        start_t, end_t = chunk_start, actual_end
        if USE_SNAP:
            ss = snap_to_low_energy(start_t, times, rms, total_duration, SNAP_WINDOW_SEC, "start")
            se = snap_to_low_energy(end_t, times, rms, total_duration, SNAP_WINDOW_SEC, "end")
            if se > ss + MIN_CHUNK_SEC:
                start_t, end_t = ss, se
        duration = end_t - start_t
        if duration > MAX_CHUNK_SEC:
            start_t, end_t = chunk_start, actual_end
            duration = end_t - start_t
        if duration < MIN_CHUNK_SEC or duration > MAX_CHUNK_SEC:
            word_idx = last_widx + 1
            continue
        start_sample, end_sample = int(start_t * sr), int(end_t * sr)
        audio = y[start_sample:end_sample]
        if len(audio) / sr < MIN_CHUNK_SEC:
            word_idx = last_widx + 1
            continue
        out_filename = source_filename.replace(".wav", f"_scaledv1_chunk{chunk_idx:04d}.wav")
        chunks.append({
            "filename": out_filename,
            "audio": audio,
            "transcription": transcription,
            "metadata": {
                "wpm": wpm,
                "audio_type": "real",
                "source_file": source_filename,
                "chunk_index": chunk_idx,
                "start_sec": round(float(start_t), 3),
                "end_sec": round(float(end_t), 3),
                "duration_sec": round(float(duration), 3),
                "header_offset": round(float(header_offset), 3),
                "pre_roll_sec": round(float(pre_roll), 3),
                "total_offset": round(float(offset), 3),
                "scale": round(float(scale), 6),
                "chunk_sec_target": CHUNK_SEC,
                "label_method": "scaled_timing_snap_silence_v1" if USE_SNAP else "scaled_timing_v1",
                "source": "W1AW",
                "farnsworth": wpm <= 15,
            },
        })
        chunk_idx += 1
        word_idx = last_widx + 1
    return chunks


def process_file(record):
    filename = record["filename"]
    try:
        wpm = record.get("metadata", {}).get("wpm") if isinstance(record.get("metadata"), dict) else None
        if wpm is None:
            wpm = get_wpm_from_filename(filename)
        wpm = int(wpm) if wpm is not None else None
        if wpm not in USE_WPMS:
            return False, {"source_file": filename, "status": "skipped_wpm", "wpm": wpm, "chunks": 0}, []
        b = get_bucket()
        raw_bytes = b.blob(f"{GCS_LABELS}/{filename.replace('.wav', '.txt')}").download_as_bytes()
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        transcription, wpm_from_label = clean_label(raw_text.strip())
        if transcription is None:
            return False, {"source_file": filename, "status": "bad_label", "wpm": wpm, "chunks": 0}, []
        transcription = filter_transcription(transcription)
        if not transcription:
            return False, {"source_file": filename, "status": "empty_transcription", "wpm": wpm, "chunks": 0}, []
        if wpm is None and wpm_from_label:
            wpm = int(wpm_from_label)
        if wpm not in USE_WPMS:
            return False, {"source_file": filename, "status": "skipped_wpm_after_label", "wpm": wpm, "chunks": 0}, []
        header_offset = calculate_header_offset(raw_text.strip(), wpm)
        with tempfile.TemporaryDirectory() as tmp:
            local_path = os.path.join(tmp, filename)
            b.blob(f"{GCS_PROCESSED}/{filename}").download_to_filename(local_path)
            y, sr = sf.read(local_path, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            total_duration = len(y) / sr
            pre_roll = detect_audio_onset(y, sr)
            offset = header_offset + pre_roll
            theory_timings = calculate_word_timings(transcription, wpm, offset)
            scaled_timings, scale, theory_dur, actual_dur = scale_word_timings(theory_timings, offset, total_duration)
            if scale is None or not scaled_timings:
                return False, {"source_file": filename, "status": "bad_scale", "wpm": wpm, "chunks": 0}, []
            if not (SCALE_MIN <= scale <= SCALE_MAX):
                return False, {
                    "source_file": filename, "status": "scale_out_of_range", "wpm": wpm, "chunks": 0,
                    "audio_duration": round(total_duration, 3), "offset": round(offset, 3),
                    "theory_content_dur": round(float(theory_dur), 3),
                    "actual_content_dur": round(float(actual_dur), 3),
                    "scale": round(float(scale), 6), "words": len(transcription.split()),
                }, []
            chunks = build_chunks(y, sr, scaled_timings, filename, wpm, offset, scale, header_offset, pre_roll)
            uploaded = []
            for ch in chunks:
                chunk_path = os.path.join(tmp, ch["filename"])
                sf.write(chunk_path, ch["audio"], sr, subtype="FLOAT")
                b.blob(f"{GCS_OUT_AUDIO}/{ch['filename']}").upload_from_filename(chunk_path)
                uploaded.append({"filename": ch["filename"], "transcription": ch["transcription"], "metadata": ch["metadata"]})
            report = {
                "source_file": filename, "status": "ok", "wpm": wpm, "chunks": len(uploaded),
                "audio_duration": round(total_duration, 3), "header_offset": round(float(header_offset), 3),
                "pre_roll_sec": round(float(pre_roll), 3), "offset": round(float(offset), 3),
                "theory_content_dur": round(float(theory_dur), 3),
                "actual_content_dur": round(float(actual_dur), 3),
                "scale": round(float(scale), 6), "words": len(transcription.split()),
                "chunk_sec": CHUNK_SEC, "snap": USE_SNAP,
            }
            return True, report, uploaded
    except Exception as e:
        return False, {"source_file": filename, "status": "error", "error": str(e), "chunks": 0}, []


def upload_json_outputs(chunked_records, reports, stats):
    bucket.blob(f"{GCS_ANNOTATIONS}/{OUT_LABELS}").upload_from_string(
        json.dumps(chunked_records, ensure_ascii=False, indent=2), content_type="application/json")
    bucket.blob(f"{GCS_ANNOTATIONS}/{OUT_REPORT}").upload_from_string(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in reports), content_type="application/jsonl")
    bucket.blob(f"{GCS_ANNOTATIONS}/{OUT_STATS}").upload_from_string(
        json.dumps(stats, ensure_ascii=False, indent=2), content_type="application/json")


def run():
    t0 = time.time()
    print("\n[1] Loading source records...")
    all_records = json.loads(bucket.blob(f"{GCS_ANNOTATIONS}/{SOURCE_JSON}").download_as_text())
    print("Total source records:", len(all_records))
    groups = defaultdict(list)
    for r in all_records:
        fname = r.get("filename", "")
        wpm = r.get("metadata", {}).get("wpm") if isinstance(r.get("metadata"), dict) else None
        if wpm is None:
            wpm = get_wpm_from_filename(fname)
        try:
            wpm = int(wpm)
        except Exception:
            continue
        if wpm in USE_WPMS:
            groups[wpm].append(r)
    selected_files = []
    print("\n[2] Source files by WPM:")
    for wpm in USE_WPMS:
        recs = groups[wpm]
        random.shuffle(recs)
        take = recs[:MAX_FILES_PER_WPM]
        selected_files.extend(take)
        print(f"WPM {wpm:>2}: available={len(recs):>5}, selected_files={len(take):>4}")
    random.shuffle(selected_files)
    print("Total selected source files:", len(selected_files))
    chunked_records, reports = [], []
    processed_files = ok_files = fail_files = 0
    idx = 0
    while idx < len(selected_files) and len(chunked_records) < TARGET_CHUNKS:
        batch = selected_files[idx:idx + BATCH_FILES]
        idx += BATCH_FILES
        print(f"\n[3] Processing batch {idx - len(batch) + 1}-{idx} / {len(selected_files)}")
        batch_t0 = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_file, rec): rec.get("filename", "") for rec in batch}
            for fut in as_completed(futures):
                ok, report, chunks = fut.result()
                processed_files += 1
                reports.append(report)
                if ok:
                    ok_files += 1
                    remaining = TARGET_CHUNKS - len(chunked_records)
                    if remaining > 0:
                        chunked_records.extend(chunks[:remaining])
                else:
                    fail_files += 1
                print(f"[{processed_files:04d}] {report.get('source_file')} | status={report.get('status')} | wpm={report.get('wpm')} | chunks={report.get('chunks')} | scale={report.get('scale')}")
        stats = {
            "target_chunks": TARGET_CHUNKS, "current_chunks": len(chunked_records),
            "processed_files": processed_files, "ok_files": ok_files, "fail_files": fail_files,
            "use_wpms": USE_WPMS, "chunk_sec": CHUNK_SEC, "min_chunk_sec": MIN_CHUNK_SEC,
            "max_chunk_sec": MAX_CHUNK_SEC, "snap": USE_SNAP,
            "scale_min": SCALE_MIN, "scale_max": SCALE_MAX,
            "output_audio": GCS_OUT_AUDIO, "output_labels": f"{GCS_ANNOTATIONS}/{OUT_LABELS}",
            "elapsed_sec": round(time.time() - t0, 2),
        }
        upload_json_outputs(chunked_records, reports, stats)
        print(f"Batch done in {time.time() - batch_t0:.1f}s | chunks={len(chunked_records)}/{TARGET_CHUNKS} | ok_files={ok_files} fail_files={fail_files}")
    by_wpm, durations = defaultdict(int), []
    for r in chunked_records:
        meta = r.get("metadata", {})
        by_wpm[str(meta.get("wpm"))] += 1
        if "duration_sec" in meta:
            durations.append(float(meta["duration_sec"]))
    stats = {
        "target_chunks": TARGET_CHUNKS, "final_chunks": len(chunked_records),
        "processed_files": processed_files, "ok_files": ok_files, "fail_files": fail_files,
        "chunks_by_wpm": dict(by_wpm),
        "duration_mean": float(np.mean(durations)) if durations else None,
        "duration_median": float(np.median(durations)) if durations else None,
        "duration_min": float(np.min(durations)) if durations else None,
        "duration_max": float(np.max(durations)) if durations else None,
        "use_wpms": USE_WPMS, "chunk_sec": CHUNK_SEC, "snap": USE_SNAP,
        "scale_min": SCALE_MIN, "scale_max": SCALE_MAX,
        "output_audio": GCS_OUT_AUDIO,
        "output_labels": f"{GCS_ANNOTATIONS}/{OUT_LABELS}",
        "output_report": f"{GCS_ANNOTATIONS}/{OUT_REPORT}",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    upload_json_outputs(chunked_records, reports, stats)
    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print("Final chunks:", len(chunked_records))
    print("Chunks by WPM:", dict(by_wpm))
    print("Duration median:", stats["duration_median"])
    print("Output labels:", f"gs://{BUCKET_NAME}/{GCS_ANNOTATIONS}/{OUT_LABELS}")
    print("Output audio:", f"gs://{BUCKET_NAME}/{GCS_OUT_AUDIO}/")
    print("Report:", f"gs://{BUCKET_NAME}/{GCS_ANNOTATIONS}/{OUT_REPORT}")
    print("Stats:", f"gs://{BUCKET_NAME}/{GCS_ANNOTATIONS}/{OUT_STATS}")


if __name__ == "__main__":
    run()
