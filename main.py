from config.settings import config
from config.logging_config import logger
from quant_alpha.data.price_loader import PriceLoader
from quant_alpha.data.fundamental_loader import FundamentalLoader

def main():
    logger.info("="*50)
    logger.info("🚀 QUANT ALPHA SYSTEM - DATA VERIFICATION")
    logger.info("="*50)
    
    # --- STEP 1: LOAD PRICES ---
    logger.info("\n--- 1. Testing PriceLoader ---")
    try:
        price_loader = PriceLoader()
        prices = price_loader.get_data()
        
        if not prices.empty:
            logger.info(f"✅ PRICES SUCCESS!")
            logger.info(f"📊 Rows: {len(prices):,}")
            logger.info(f"📅 Range: {prices['date'].min()} to {prices['date'].max()}")
        else:
            logger.error("❌ Prices DataFrame is empty!")
    except Exception as e:
        logger.exception(f"❌ Price Loader Failed: {e}")

    # --- STEP 2: LOAD FUNDAMENTALS (GOLDMINE CHECK) ---
    logger.info("\n--- 2. Testing FundamentalLoader ---")
    try:
        fund_loader = FundamentalLoader()
        # Force reload to ensure we get the NEW columns, not the old cached file
        funds = fund_loader.get_data(force_reload=True) 
        
        if not funds.empty:
            logger.info(f"✅ FUNDAMENTALS SUCCESS!")
            logger.info(f"📊 Stocks Found: {len(funds):,}")
            
            # Print New Metrics to Console
            print("\nSample Data (The Goldmine):")
            # Hum naye columns check kar rahe hain: ROE, Debt, EPS
            cols_to_show = ['ticker', 'sector', 'roe', 'debt_to_equity', 'eps', 'fcf']
            
            # Sirf wahi columns dikhao jo exist karte hain
            valid_cols = [c for c in cols_to_show if c in funds.columns]
            print(funds[valid_cols].head(5))
            
            # Verify specific columns exist
            if 'roe' in funds.columns and 'eps' in funds.columns:
                 logger.info("✨ SUCCESS: Advanced Metrics (ROE, EPS) Detected!")
            else:
                 logger.warning("⚠️ WARNING: Advanced Metrics missing!")
                 
        else:
            logger.error("❌ Fundamentals DataFrame is empty!")
    except Exception as e:
        logger.exception(f"❌ Fundamental Loader Failed: {e}")

    logger.info("\n" + "="*50)
    logger.info("🎉 DATA ENGINE TEST COMPLETE")
    logger.info("="*50)

if __name__ == "__main__":
    main()