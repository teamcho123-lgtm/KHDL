"""Data Manifest Preparation Script for SpeechBrain.
Scans raw dataset folders (specifically RAVDESS),
extracts acoustic physical features with Parselmouth,
maps labels to binary stress classes (normal vs stress),
and creates balanced speaker-independent train.json, valid.json, test.json.
"""

import os
import json
import glob
import argparse
from typing import Dict, List, Any
import soundfile as sf
import numpy as np
import sys
from tqdm import tqdm

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.acoustic_extractor import AcousticExtractor


def parse_ravdess_filename(filename: str, include_all: bool = False) -> Dict[str, Any]:
    """RAVDESS format: 03-01-XX-01-01-01-YY.wav
    XX is emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    YY is actor id: 01 to 24 (even: female, odd: male)
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split("-")
    if len(parts) != 7:
        return {}

    emotion_code = parts[2]
    actor_id_num = parts[6]
    actor_code = f"Actor_{actor_id_num}"

    # Binary Stress Mapping:
    # 01 (Neutral), 02 (Calm) -> normal
    # 05 (Angry), 06 (Fearful) -> stress
    if include_all:
        stress_mapping = {
            "01": "normal",
            "02": "normal",
            "03": "normal",
            "04": "stress", # Sad is low-energy stress / depressive affect
            "05": "stress", # Angry is high-energy stress
            "06": "stress", # Fearful is high-arousal stress
            "07": "stress", # Disgust is negative valence
        }
    else:
        # Standard high-contrast benchmark (Neutral/Calm vs Angry/Fearful)
        stress_mapping = {
            "01": "normal",
            "02": "normal",
            "05": "stress",
            "06": "stress",
        }

    if emotion_code not in stress_mapping:
        return {}

    return {
        "speaker_id": actor_code,
        "emotion_code": emotion_code,
        "stress_label": stress_mapping[emotion_code],
    }


def prepare_manifests(
    raw_dir: str = "data/raw/RAVDESS",
    output_dir: str = "data/manifests",
    include_all: bool = False,
):
    """Generates balanced speaker-independent train.json, valid.json, test.json."""
    os.makedirs(output_dir, exist_ok=True)
    extractor = AcousticExtractor()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 24 Actors split into balanced Gender groups (odd: male, even: female):
    # Test set (4 actors: 2 male, 2 female)
    test_speakers = ["Actor_21", "Actor_22", "Actor_23", "Actor_24"]
    # Valid set (4 actors: 2 male, 2 female)
    valid_speakers = ["Actor_17", "Actor_18", "Actor_19", "Actor_20"]
    # Train set (16 actors: 8 male, 8 female: Actor_01 to Actor_16)

    wav_files = glob.glob(os.path.join(raw_dir, "**", "*.wav"), recursive=True)
    # Exclude any test_samples folder
    wav_files = [w for w in wav_files if "test_samples" not in w]

    print(f"[ManifestBuilder] Found {len(wav_files)} candidate wav files in {raw_dir}")
    if len(wav_files) == 0:
        print(f"Warning: No audio files found in {raw_dir}. Please run data/download_ravdess.py first!")
        return

    manifest_train = {}
    manifest_valid = {}
    manifest_test = {}

    skipped = 0
    print("[ManifestBuilder] Pre-computing physical acoustic features (F0, Jitter, Shimmer, HNR)...")
    for wav_path in tqdm(wav_files, desc="Extracting features"):
        meta = parse_ravdess_filename(wav_path, include_all=include_all)
        if not meta:
            skipped += 1
            continue

        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        speaker_id = meta["speaker_id"]
        stress_label = meta["stress_label"]

        info = sf.info(wav_path)
        duration = float(info.duration)

        # Pre-extract physical features
        try:
            phys_vec = extractor.get_feature_vector(wav_path).tolist()
        except Exception as e:
            skipped += 1
            print(f"[ManifestBuilder] Skipping {wav_path}: {e}")
            continue
        if not all(np.isfinite(value) for value in phys_vec) or phys_vec[0] <= 0:
            skipped += 1
            print(f"[ManifestBuilder] Skipping unusable voice sample: {wav_path}")
            continue

        entry = {
            "wav": os.path.relpath(os.path.abspath(wav_path), project_root),
            "duration": round(duration, 2),
            "speaker_id": speaker_id,
            "acoustic_phys": phys_vec,
            "stress_label": stress_label,
        }

        # Speaker-independent partition
        if speaker_id in test_speakers:
            manifest_test[utt_id] = entry
        elif speaker_id in valid_speakers:
            manifest_valid[utt_id] = entry
        else:
            manifest_train[utt_id] = entry

    # Save to JSON
    for name, data in [("train.json", manifest_train), ("valid.json", manifest_valid), ("test.json", manifest_test)]:
        out_path = os.path.join(output_dir, name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[ManifestBuilder] Saved {len(data)} items to {out_path}")

    print(f"Summary: Processed {len(manifest_train) + len(manifest_valid) + len(manifest_test)} files, Skipped {skipped} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SpeechBrain JSON manifests from RAVDESS")
    parser.add_argument("--raw_dir", default="data/raw/RAVDESS", help="Path to raw audio folder")
    parser.add_argument("--out_dir", default="data/manifests", help="Output directory for manifests")
    parser.add_argument("--all_emotions", action="store_true", help="Include all emotions instead of high-contrast only")
    args = parser.parse_args()
    prepare_manifests(args.raw_dir, args.out_dir, include_all=args.all_emotions)
