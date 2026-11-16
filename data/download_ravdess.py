"""Downloader & unpacker for the RAVDESS Speech Dataset from Zenodo."""

import os
import urllib.request
import zipfile
import glob
import tempfile
from pathlib import Path
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
DEST_ZIP = "data/raw/Audio_Speech_Actors_01-24.zip"
EXTRACT_DIR = "data/raw/RAVDESS"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # 1. Download zip file if not already present
    if not os.path.exists(DEST_ZIP):
        print(f"[Downloader] Downloading RAVDESS dataset from Zenodo (~200MB)...")
        descriptor, temporary_zip = tempfile.mkstemp(
            prefix="ravdess_", suffix=".zip", dir=os.path.dirname(DEST_ZIP)
        )
        os.close(descriptor)
        try:
            with DownloadProgressBar(
                unit="B", unit_scale=True, miniters=1, desc="RAVDESS Download"
            ) as t:
                urllib.request.urlretrieve(
                    ZENODO_URL,
                    filename=temporary_zip,
                    reporthook=t.update_to,
                )
            with zipfile.ZipFile(temporary_zip, "r") as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise ValueError(f"Downloaded archive is corrupt at {bad_member}")
            os.replace(temporary_zip, DEST_ZIP)
        finally:
            if os.path.exists(temporary_zip):
                os.remove(temporary_zip)
        print(f"[Downloader] Download completed: {DEST_ZIP}")
    else:
        print(f"[Downloader] Found existing archive at {DEST_ZIP}")

    # 2. Extract zip archive
    print(f"[Downloader] Extracting archive to {EXTRACT_DIR}...")
    with zipfile.ZipFile(DEST_ZIP, "r") as zip_ref:
        bad_member = zip_ref.testzip()
        if bad_member is not None:
            raise ValueError(f"RAVDESS archive is corrupt at {bad_member}")
        extraction_root = Path(EXTRACT_DIR).resolve()
        for member in zip_ref.infolist():
            safe_name = member.filename.replace("\\", "/")
            destination = (extraction_root / safe_name).resolve()
            if destination != extraction_root and extraction_root not in destination.parents:
                raise ValueError(f"Unsafe path in RAVDESS archive: {member.filename}")
        zip_ref.extractall(EXTRACT_DIR)

    # Verify extracted wav files
    wav_files = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.wav"), recursive=True)
    actor_folders = glob.glob(os.path.join(EXTRACT_DIR, "Actor_*"))
    print(f"[Downloader] Verification:")
    print(f"   - Actor directories: {len(actor_folders)} (Expected: 24)")
    print(f"   - Total .wav files: {len(wav_files)} (Expected: 1440)")
    if len(wav_files) != 1440 or len(actor_folders) != 24:
        raise RuntimeError(
            "Unexpected RAVDESS archive contents: "
            f"found {len(actor_folders)} actor folders and {len(wav_files)} WAVs"
        )
    print(f"✅ RAVDESS dataset ready at: {EXTRACT_DIR}")


if __name__ == "__main__":
    download_and_extract()
