# model_loader.py (updated)

from pathlib import Path
import gdown

# Create a models directory if it doesn't exist
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILENAME = "waste_classifier_model.h5"
MODEL_FILE_ID = "1g2_CARyq04EK1TqlrpmGEfNtDedhXWoN"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

def download_model_from_gdrive(file_id: str, dest_path: Path = MODEL_PATH):
    """
    Download the model from Google Drive if not already present.
    Returns the path to the downloaded model.
    """
    if dest_path.exists():
        print(f"Model already exists at {dest_path}")
        return str(dest_path)

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading model from Google Drive: {url}")
    gdown.download(url, str(dest_path), quiet=False, fuzzy=True)

    if not dest_path.exists():
        raise RuntimeError("Model download failed.")
    
    print(f"Model downloaded successfully to {dest_path}")
    return str(dest_path)

def get_model():
    """
    Lazy-load and return the Keras model.
    """
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    model_file = download_model_from_gdrive(MODEL_FILE_ID)
    print("Loading model...")
    model = load_model(model_file)
    print("Model loaded successfully!")
    return model
