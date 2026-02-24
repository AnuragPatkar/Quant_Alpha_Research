"""
FACTOR CLUSTER ANALYSIS
Checks if selected features are orthogonal (diverse) or clustered (redundant).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import logging
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
CACHE_DATA_PATH = r"E:\coding\quant_alpha_research\data\cache\master_data_with_factors.parquet"
MODEL_PATH = r"E:\coding\quant_alpha_research\models\production\lightgbm_latest.pkl"

def analyze_clusters():
    if not os.path.exists(CACHE_DATA_PATH):
        logger.error("❌ Master data cache not found.")
        return
    
    if not os.path.exists(MODEL_PATH):
        logger.error("❌ Production model not found. Run training first.")
        return

    # 1. Load Feature Names from Model
    logger.info("📦 Loading Production Model...")
    payload = joblib.load(MODEL_PATH)
    features = payload.get('feature_names', [])
    
    if not features:
        logger.error("❌ No features found in model payload.")
        return
        
    logger.info(f"✅ Found {len(features)} selected features.")

    # 2. Load Data (Sample for Correlation)
    logger.info("📊 Loading Data for Correlation Analysis...")
    df = pd.read_parquet(CACHE_DATA_PATH, columns=features)
    
    # Sample last 1 year for relevant correlation
    df = df.tail(252 * 2000) # Approx 1 year for 2000 stocks
    
    # 3. Calculate Correlation Matrix
    logger.info("🧮 Calculating Correlation Matrix...")
    
    # Fix: Filter out categorical columns (strings) that cause correlation to fail
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        logger.warning("⚠️ Need at least 2 numeric features to perform clustering.")
        return

    corr_matrix = numeric_df.corr(method='spearman').fillna(0)
    
    # 4. Cluster Analysis
    # Convert correlation to distance (1 - |corr|)
    distance_matrix = 1 - np.abs(corr_matrix)
    linkage = hierarchy.linkage(squareform(distance_matrix), method='ward')
    
    # 5. Visualize
    plt.figure(figsize=(12, 10))
    dendro = hierarchy.dendrogram(
        linkage, 
        labels=corr_matrix.columns, 
        leaf_rotation=90, 
        leaf_font_size=8
    )
    plt.title(f"Feature Clustering Dendrogram ({len(features)} Features)")
    plt.tight_layout()
    plt.show()
    
    # 6. Report High Correlations
    logger.info("\n⚠️  HIGH CORRELATION PAIRS (> 0.7):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                high_corr_pairs.append(pair)
                print(f"  - {pair[0]} <--> {pair[1]}: {pair[2]:.2f}")
                
    if not high_corr_pairs:
        logger.info("  ✅ No highly correlated pairs found. Features are orthogonal.")

if __name__ == "__main__":
    analyze_clusters()