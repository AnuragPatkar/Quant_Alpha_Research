from config.settings import config
from config.logging_config import logger
from quant_alpha.data.price_loader import PriceLoader
from quant_alpha.data.fundamental_loader import FundamentalLoader

def main():
    logger.info("="*50)
    logger.info("🚀 QUANT ALPHA SYSTEM INITIALIZATION")
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
            logger.info(f"🏢 Tickers: {prices['ticker'].nunique()}")
        else:
            logger.error("❌ Prices DataFrame is empty!")
    except Exception as e:
        logger.exception(f"❌ Price Loader Failed: {e}")
    

    # --- STEP 2: LOAD FUNDAMENTALS ---
    logger.info("\n--- 2. Testing FundamentalLoader ---")
    try:
        fund_loader = FundamentalLoader()
        funds = fund_loader.get_data()
        
        if not funds.empty:
            logger.info(f"✅ FUNDAMENTALS SUCCESS!")
            logger.info(f"📊 Stocks Found: {len(funds):,}")
            logger.info(f"🏢 Sectors Found: {funds['sector'].nunique()}")
            
            # Print Sector Distribution to Console
            print("\nSector Distribution (Top 5):")
            print(funds['sector'].value_counts().head())
            
            print("\nSample Data:")
            print(funds[['ticker', 'sector', 'market_cap', 'pe_ratio']].head(3))
        else:
            logger.error("❌ Fundamentals DataFrame is empty!")
    except Exception as e:
        logger.exception(f"❌ Fundamental Loader Failed: {e}")

    logger.info("\n" + "="*50)
    logger.info("🎉 DATA ENGINE TEST COMPLETE")
    logger.info("="*50)

if __name__ == "__main__":
    main()









