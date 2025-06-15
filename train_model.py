"""
Book Genre Predictor - Model Training Module

This module handles the training and evaluation of the book genre prediction model.
It includes functions for feature engineering, model training, and evaluation.

Key functions:
- prepare_features: Prepares features for model training
- train_model: Trains the genre prediction model
- evaluate_model: Evaluates model performance
- save_model: Saves the trained model and vectorizer
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_features(df):
    """
    Prepare features for model training.
    
    Args:
        df (pd.DataFrame): Input dataframe with cleaned data
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    try:
        # Combine text features
        df['text_features'] = (
            df['clean_title'] + ' ' + 
            df['clean_authors'] + ' ' + 
            df['clean_top_tags']
        ).fillna('').astype(str) # Ensure no NaN and convert to string
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Transform text features
        X_text = vectorizer.fit_transform(df['text_features'])
        
        # Prepare numerical features
        numerical_features = [
            'rating_std', 'rating_skew', 'high_rating_ratio',
            'popularity', 'rating_score', 'engagement_score',
            'title_length', 'author_count', 'tag_count'
        ]
        X_num = df[numerical_features].values
        
        # Combine features
        X = np.hstack([X_text.toarray(), X_num])
        y = df['genre_encoded']
        
        return X, y, vectorizer
        
    except Exception as e:
        logger.error(f"Error in feature preparation: {str(e)}")
        return None, None, None

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train the genre prediction model.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (model, X_test, y_test) or (None, None, None) if error
    """
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Initialize and train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
        
        logger.info("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info(f"Training accuracy: {train_score:.3f}")
        logger.info(f"Testing accuracy: {test_score:.3f}")
        
        return model, X_test, y_test
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        return None, None, None

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test (np.ndarray): Test feature matrix
        y_test (np.ndarray): Test target vector
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    try:
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Log results
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        return None

def save_model(model, vectorizer, model_path='book_genre_model.joblib', 
              vectorizer_path='tfidf_vectorizer.joblib'):
    """
    Save the trained model and vectorizer.
    
    Args:
        model: Trained model
        vectorizer: Fitted TF-IDF vectorizer
        model_path (str): Path to save model
        vectorizer_path (str): Path to save vectorizer
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False

def main():
    """Main function to run the model training pipeline."""
    try:
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv('cleaned_books_genre.csv')
        
        # Prepare features
        logger.info("Preparing features...")
        X, y, vectorizer = prepare_features(df)
        if X is None:
            return
        
        # Train model
        logger.info("Training model...")
        model, X_test, y_test = train_model(X, y)
        if model is None:
            return
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        if metrics is None:
            return
        
        # Save model
        logger.info("Saving model...")
        if not save_model(model, vectorizer):
            return
        
        logger.info("Model training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")

if __name__ == "__main__":
    main() 