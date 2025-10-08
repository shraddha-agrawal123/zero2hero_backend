# model_loader.py (only relevant parts)

from pathlib import Path
import gdown

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILENAME = "waste_classifier_model.h5"
MODEL_FILE_ID = "1g2_CARyq04EK1TqlrpmGEfNtDedhXWoN"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

def download_model_from_gdrive(file_id: str, dest_path: Path = MODEL_PATH):
    if dest_path.exists():
        return str(dest_path)
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    gdown.download(url, str(dest_path), quiet=False)
    if not dest_path.exists():
        raise RuntimeError("Download failed.")
    return str(dest_path)

def get_model():
    # lazy-import tensorflow and load model only when called
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    model_file = download_model_from_gdrive(MODEL_FILE_ID)
    model = load_model(model_file)
    return model
