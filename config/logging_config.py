import logging
import sys
from pathlib import Path

# Add project root to path
# sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config

def setup_logging(name='Quant_Alpha'):
    """
    Sets up a logger that writes to both:
    1. Console (Terminal) - For quick checks
    2. File (logs/quant_alpha_ENV.log) - For permanent record
    """

    # Get log files path from settings
    log_file = config.LOG_FILE

    # 1. Define Format (Time - Level - Message)
    log_format = logging.Formatter(config.LOG_FORMAT,datefmt=config.LOG_DATE_FORMAT)

    # Determine Log Level from Config
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    # 2. File Handler (Saves to disk)
    file_handler = logging.FileHandler(log_file,mode='a',encoding='utf-8')
    file_handler.setFormatter(log_format)
    file_handler.setLevel(log_level)

    # 3. Console Handler (Shows in terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(log_level)  

    # 4. Setup the logger object 
    logger = logging.getLogger(name)
    
    # Set global level based on Config (DEBUG/INFO/WARNING)
    logger.setLevel(log_level)

    # Avoid adding handlers multiple times (Duplicate log prevention)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Create a default logger instance to import easily
logger = setup_logging()
