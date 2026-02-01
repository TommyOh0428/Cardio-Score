import logging

def get_logger():
    # Configure logging
    logging.basicConfig(level=logging.INFO, filename='training.log', filemode='w',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("CardioScore")
    return logger

logger = get_logger()
