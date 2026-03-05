"""
main.py
=======
Project Entry Point
Wraps scripts/run_pipeline.py for easy execution.
"""
import sys
from pathlib import Path

# Add 'scripts' to path to import run_pipeline
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.append(str(SCRIPTS_DIR))

try:
    from scripts.run_pipeline import run
except ImportError as e:
    print("❌ Error: Could not import scripts/run_pipeline.py")
    print(f"   Details: {e}")
    sys.exit(1)

if __name__ == "__main__":
    run()