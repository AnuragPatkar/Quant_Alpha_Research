"""Test data loading"""
import sys
from pathlib import Path

# Need to add project root to path so imports work
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from quant_alpha.data import DataLoader

print("Testing DataLoader...")
loader = DataLoader()
data = loader.load()
print(f"\n✓ Data loaded successfully: {len(data)} rows, {data['ticker'].nunique()} stocks")
print(f"✓ Date range: {data['date'].min()} to {data['date'].max()}")
print(f"✓ Columns: {list(data.columns)}")
