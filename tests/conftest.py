import sys
from pathlib import Path

# Plotting/validation scripts (validate_study_run.py, recompute_geometry.py,
# context_revelation_trajectory.py, etc.) live outside the repo in the
# workspace-level visualisations/ directory, sibling to this project.
VIS_DIR = Path(__file__).resolve().parents[2] / "visualisations"
if VIS_DIR.is_dir():
    sys.path.insert(0, str(VIS_DIR))
