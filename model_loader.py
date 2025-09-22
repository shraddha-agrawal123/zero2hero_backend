# model_loader.py

import os
from pathlib import Path
import gdown
from tensorflow.keras.models import load_model

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Hardcoded filename and Google Drive file ID
MODEL_FILENAME = "waste_classifier_model.h5"
MODEL_FILE_ID = "1g2_CARyq04EK1TqlrpmGEfNtDedhXWoN"  # Your shared file ID
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

def download_model_from_gdrive(file_id: str, dest_path: Path = MODEL_PATH):
    """Download file from Google Drive if not already present."""
    if dest_path.exists():
        print(f"[model_loader] Model already exists at {dest_path}")
        return str(dest_path)

    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    print(f"[model_loader] Downloading model from {url} to {dest_path} ...")
    gdown.download(url, str(dest_path), quiet=False)
    if not dest_path.exists():
        raise RuntimeError("[model_loader] Download failed: model file not found.")
    return str(dest_path)

def get_model():
    model_file = download_model_from_gdrive(MODEL_FILE_ID)
    print("[model_loader] Loading model ...")
    model = load_model(model_file)
    print("[model_loader] Model loaded successfully.")
    return model