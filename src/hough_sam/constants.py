from pathlib import Path

# Project root is two levels up from this file (src/hough_sam)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRAMES_DIR = PROJECT_ROOT / "frames"
SEG_RESULTS_DIR = PROJECT_ROOT / "seg_results"
FIGURES_DIR = PROJECT_ROOT / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"
SEG_VIDEOS_DIR = PROJECT_ROOT / "seg_videos_2"
VIDEO_FEATURES_DIR = PROJECT_ROOT / "video_features"

# labels file per user instruction
LABELS_FILE = PROJECT_ROOT / "labels" / "JP wound ratings.xlsx"
