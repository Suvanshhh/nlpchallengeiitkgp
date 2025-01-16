import os

class Config:
    # Base paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
    
    # Model paths
    GEMMA_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'Gemma_LoRA_finetuned_weights.h5')
    SENTIMENT_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'Sentiment_Model')
    CLASSIFIER_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'Classifier_model')
    
    # Flask configurations
    DEBUG = True
    PORT = 5000
    HOST = 'localhost'