from pathlib import Path
import logging
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Overall
    ROOT_PATH: Path = Path(__file__).resolve().parent

    # Folder/paths
    MODELS_FOLDER: Path = ROOT_PATH / "models"
    CLASSES_FILE = MODELS_FOLDER / "classes.txt"
    VIDOES_FOLDER: Path = ROOT_PATH / "videos"

    # Model params
    IMG_SIZE: int  = 416
    NUM_BOXES: int = 1
    NUM_CLASSES: int = 1
    WIDTH_MULT: int = .7
    NUM_ANCHORS: int = 3

    # LOGGER
    LOG_FOLDER: Path = ROOT_PATH / "logs"
    CLI_LOG_LEVEL: int = logging.INFO
    FILE_LOG_LEVEL: int = logging.INFO

    ANCHORS_PER_SCALE: list[list[tuple[float, float]]] = [
        # Scale 13×13 (large objects)
        [(0.2100, 0.5531), (0.4212, 0.4984), (0.5946, 0.8193)],
        # Scale 26×26 (medium objects)
        [(0.1297, 0.1785), (0.1175, 0.4068), (0.2801, 0.2911)],
        # Scale 52×52 (small objects)
        [(0.0268, 0.0622), (0.0441, 0.1462), (0.0580, 0.2430)],
    ]

