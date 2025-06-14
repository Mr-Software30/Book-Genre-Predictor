"""
Book Genre Predictor - Model Training Module

This module handles the training of the genre prediction model using Random Forest Classifier.
It includes feature engineering, model training, evaluation, and saving the trained model.

Key functions:
- prepare_features: Prepares text and numerical features for model training
- train_model: Trains and evaluates the Random Forest model
- save_model: Saves the trained model and vectorizer
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

def prepare_features(df):
    """
    Prepare features for model training.
    
    Args:
        df (DataFrame): Input DataFrame with processed data
        
    Returns:
        tuple: (X_text, X_numerical, y) where:
            - X_text: TF-IDF features from text
            - X_numerical: Numerical features
            - y: Target labels
    """
    # Prepare text features
    vectorizer = TfidfVectorizer(max_features=1000)
    X_text = vectorizer.fit_transform(df['content_features'])
    
    # Prepare numerical features
    numerical_features = ['popularity', 'rating_score', 'engagement_score']
    X_numerical = df[numerical_features].values
    
    # Prepare target
    y = df['genre_encoded'].values
    
    return X_text, X_numerical, y, vectorizer

def train_model(X_text, X_numerical, y):
    """
    Train the Random Forest model.
    
    Args:
        X_text: TF-IDF features
        X_numerical: Numerical features
        y: Target labels
        
    Returns:
        tuple: (trained_model, accuracy) where:
            - trained_model: The trained Random Forest model
            - accuracy: Model accuracy on test set
    """
    # Combine features
    X = np.hstack([X_text.toarray(), X_numerical])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy

def save_model(model, vectorizer, accuracy):
    """
    Save the trained model and vectorizer.
    
    Args:
        model: Trained Random Forest model
        vectorizer: TF-IDF vectorizer
        accuracy: Model accuracy
    """
    # Save model
    joblib.dump(model, 'book_genre_model.joblib')
    
    # Save vectorizer
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    
    print(f"\nModel saved with accuracy: {accuracy:.2%}")

def main():
    """
    Main function to run the model training pipeline.
    """
    print("Loading data...")
    df = pd.read_csv('cleaned_books_genre.csv')
    
    print("Preparing features...")
    X_text, X_numerical, y, vectorizer = prepare_features(df)
    
    print("Training model...")
    model, accuracy = train_model(X_text, X_numerical, y)
    
    print("Saving model...")
    save_model(model, vectorizer, accuracy)

if __name__ == "__main__":
    main() 