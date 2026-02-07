from config.settings import config
from config.logging_config import logger
from quant_alpha.data.price_loader import PriceLoader

def main():
    logger.info("="*50)
    logger.info("🚀 QUANT ALPHA SYSTEM INITIALIZATION")
    logger.info("="*50)

    # 1. Initialize Loader
    logger.info("🛠️ Initializing PriceLoader...")
    loader = PriceLoader()

    # 2. Load Data (This will triger Cache Check -> Load raw -> Save Cache)
    try:
        df = loader.get_data()

        # 3. Show Results
        if not df.empty:
            logger.info("\n" + "="*50)
            logger.info(f"✅ SUCCESS! Data Loaded.")
            logger.info(f"📊 Total Rows: {len(df):,}")
            logger.info(f"📅 Date Range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"🏢 Unique Tickers: {df['ticker'].nunique()}")
            logger.info("="*50)

            # print head strictly to console (not log file) for readability
            print("\nFirst 5 Rows:")
            print(df.head())
        else:
            logger.error("❌ Data Load Failed: DataFrame is empty.")

    except Exception as e:
        logger.exception(f"❌ Critical Error in Main Loop: {e}")

if __name__ == "__main__":
    main()









