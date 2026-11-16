"""SpeechBrain DataIO Pipeline & Dynamic Batching.
Loads manifests into DynamicItemDataset with on-the-fly audio reading,
physical acoustic feature conversion, label encoding, and DynamicBatchSampler.
"""

import os
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
import speechbrain as sb
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.dataio.batch import PaddedBatch


def dataio_prep(hparams: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """Prepares DynamicItemDatasets for train, valid, and test sets.
    Configures:
    1. sig: audio waveform loaded on-the-fly via read_audio.
    2. acoustic_phys_tensor: 5-dim physical vocal perturbation tensor.
    3. stress_label_encoded: categorical index (0: normal, 1: stress).
    """
    train_json = hparams["train_json"]
    valid_json = hparams["valid_json"]
    test_json = hparams["test_json"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=train_json)
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=valid_json)
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=test_json)

    datasets = [train_data, valid_data, test_data]

    sample_rate = int(hparams.get("sample_rate", 16000))
    project_root = Path(__file__).resolve().parents[1]

    # Normalize acoustic features using train-set statistics only. Persist the
    # values so inference applies exactly the same transform.
    with open(train_json, "r", encoding="utf-8") as f:
        train_annotations = json.load(f)
    raw_acoustic = np.asarray(
        [item["acoustic_phys"] for item in train_annotations.values()],
        dtype=np.float32,
    )
    if raw_acoustic.ndim != 2 or raw_acoustic.shape[1] != 5:
        raise ValueError("Training manifest acoustic_phys must contain 5 values per item")
    if not np.isfinite(raw_acoustic).all():
        raise ValueError("Training manifest contains non-finite acoustic features")
    acoustic_mean = raw_acoustic.mean(axis=0)
    acoustic_std = raw_acoustic.std(axis=0)
    acoustic_std[acoustic_std < 1e-6] = 1.0

    save_folder = hparams.get("save_folder", "results/checkpoints")
    os.makedirs(save_folder, exist_ok=True)
    normalization_path = os.path.join(save_folder, "acoustic_normalization.npz")
    np.savez(normalization_path, mean=acoustic_mean, std=acoustic_std)
    acoustic_mean_tensor = torch.tensor(acoustic_mean, dtype=torch.float32)
    acoustic_std_tensor = torch.tensor(acoustic_std, dtype=torch.float32)

    # 1. Audio waveform pipeline: RAVDESS is stored at 48 kHz; Wav2Vec2
    # expects 16 kHz. Downmix and resample every item before batching.
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        audio_path = Path(wav)
        if not audio_path.is_absolute():
            audio_path = project_root / audio_path
        audio, original_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        signal = torch.from_numpy(audio.mean(axis=1).copy())
        if signal.numel() == 0:
            raise ValueError(f"Audio file is empty: {wav}")
        if original_rate != sample_rate:
            signal = torchaudio.functional.resample(
                signal, orig_freq=original_rate, new_freq=sample_rate
            )
        return signal

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 2. Acoustic physical features tensor pipeline (5-dimensional: F0, F0_std, Jitter, Shimmer, HNR)
    @sb.utils.data_pipeline.takes("acoustic_phys")
    @sb.utils.data_pipeline.provides("acoustic_phys_tensor")
    def acoustic_pipeline(acoustic_phys):
        features = torch.tensor(acoustic_phys, dtype=torch.float32)
        return (features - acoustic_mean_tensor) / acoustic_std_tensor

    sb.dataio.dataset.add_dynamic_item(datasets, acoustic_pipeline)

    # 3. Stress label encoder pipeline (normal: 0, stress: 1)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.enforce_label("normal", 0)
    label_encoder.enforce_label("stress", 1)

    @sb.utils.data_pipeline.takes("stress_label")
    @sb.utils.data_pipeline.provides("stress_label", "stress_label_encoded")
    def label_pipeline(stress_label):
        yield stress_label
        label_encoded = label_encoder.encode_label_torch(stress_label)
        yield label_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # Persist the canonical mapping. Older generated files in this project
    # used the reverse order, which made model outputs easy to misinterpret.
    encoder_path = os.path.join(save_folder, "label_encoder.txt")
    labels = {item.get("stress_label") for item in train_annotations.values()}
    if labels != {"normal", "stress"}:
        raise ValueError(
            "Training manifest must contain exactly the 'normal' and 'stress' labels"
        )
    label_encoder.save(encoder_path)

    # 4. Set final output keys expected by the Brain compute_forward
    output_keys = ["id", "sig", "duration", "acoustic_phys_tensor", "stress_label_encoded"]
    sb.dataio.dataset.set_output_keys(datasets, output_keys)

    return train_data, valid_data, test_data, label_encoder


def create_dynamic_dataloader(
    dataset: Any,
    max_batch_length: float = 25.0,
    num_buckets: int = 10,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Creates a DataLoader using SpeechBrain's DynamicBatchSampler.
    Batches audio clips by similar duration up to max_batch_length seconds.
    Minimizes zero-padding and GPU/CPU RAM waste.
    """
    sampler = DynamicBatchSampler(
        dataset,
        max_batch_length=max_batch_length,
        num_buckets=num_buckets,
        length_func=lambda x: x["duration"],
        shuffle=shuffle,
        batch_ordering="random" if shuffle else "ascending",
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=PaddedBatch,
        num_workers=num_workers,
    )
    return dataloader
