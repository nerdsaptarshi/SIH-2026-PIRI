import os
from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent.parent
MODEL_DIR=Path(os.getenv("MODEL_DIR",str(BASE_DIR/"models")))
MODEL_DIR.mkdir(parents=True,exist_ok=True)
