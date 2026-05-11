from pathlib import Path
import sys

file_path = Path(__file__).resolve()
root_path = file_path.parent

if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path.relative_to(Path.cwd())

IMAGE = 'Image'
VIDEO = 'Video'
SOURCES_LIST = [IMAGE, VIDEO]

DEFAULT_IMAGE = 'gun1.jpg'
DETECTED_IMAGE = 'detected_image.jpg'

DETECTION_MODEL = 'BestFinal.pt'
