# HoughSAM: Geometry-Guided Zero-Shot Segmentation of Main Incision in Cataract Surgery Using SAM2 for Skill Assessment

This repository contains code and notebooks for the Hough-SAM project. 

Layout
- `src/hough_sam/` - Python package containing modularized code
  - `segmentation/` - SAM helpers, Hough ring detection, visualization, postprocessing
  - `modeling/` - data loading, feature processing, training utilities
  - `hough_sam_segmentation_pipeline.py` - segmentation pipeline script
  - `qualitative_baseline.py` - baseline visualization script
  - `surgical_skill_prediction.py` - training/prediction script for case-level labels
- `python_notebooks/` - renamed notebook copies pointing to the new scripts

Important paths (project-root relative):
- `frames/` - input frames
- `seg_results/` - segmentation outputs
- `figures/` - saved figures
- `labels/JP wound ratings.xlsx` - label spreadsheet (referenced by code)

How to run
1. Install dependencies (OpenCV, scikit-learn, pytorch if needed for SAM):
   - pip install -r requirements.txt  # (create if needed)
2. Run scripts from repository root:
```
python -m src.hough_sam.hough_sam_segmentation_pipeline
python -m src.hough_sam.qualitative_baseline
python -m src.hough_sam.surgical_skill_prediction
```
