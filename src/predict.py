from src.train_xgb import train_xgboost
from src.utils import logger

def predict():
    # logging config is now handled in utils
    logger.info("Prediction started.")
    train_xgboost()
    logger.info("Prediction finished.")


if __name__ == "__main__":
    predict()