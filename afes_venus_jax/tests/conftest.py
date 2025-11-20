import sys
from pathlib import Path

# Ensure repository root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
