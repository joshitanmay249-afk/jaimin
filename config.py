"""
config.py
---------
Database credentials — edit DB_PASSWORD if yours is different.
"""

DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "password*1710",
    "database": "handwriting_db",
}

# Path to the trained scikit-learn model
MODEL_PATH = "model_mnist_28x28.joblib"
