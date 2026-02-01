from src.train_xgb import train_xgboost
from src.train_lr import train_logistic_regression
from src.utils import logger

def predict():
    # logging config is now handled in utils
    logger.info("Prediction started.")
    
    # Train both models
    logger.info("\n" + "="*60)
    logger.info("TRAINING LOGISTIC REGRESSION")
    logger.info("="*60)
    train_logistic_regression()
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING XGBOOST")
    logger.info("="*60)
    train_xgboost()
    
    logger.info("Prediction finished.")


if __name__ == "__main__":
    predict()