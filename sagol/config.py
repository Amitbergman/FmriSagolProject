from pathlib import Path

import logbook

ROOT_DIR = Path(__file__).parent

ROIS_DIR = ROOT_DIR.parent / 'rois'

LOGGING_LEVEL = logbook.DEBUG

# Can be `z-score` or `zero-one`.
NORMALIZATION = 'z-score'
