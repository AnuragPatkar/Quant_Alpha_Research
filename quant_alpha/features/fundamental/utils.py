import pandas as pd
from typing import Optional
from config.mappings import COLUMN_MAPPINGS

class FundamentalColumnValidator:
    """
    Centralized Knowledge Base for Column Mapping.
    Matches 'fundamentals.parquet' schema.
    """
    @classmethod
    def find_column(cls, df: pd.DataFrame, key: str) -> Optional[str]:
        # 1. Direct Check
        if key in df.columns: return key
        # 2. Mapping Check
        if key in COLUMN_MAPPINGS:
            for variant in COLUMN_MAPPINGS[key]:
                if variant in df.columns: return variant
        # 3. Case-Insensitive
        col_map_lower = {c.lower(): c for c in df.columns}
        if key.lower() in col_map_lower: return col_map_lower[key.lower()]
        return None