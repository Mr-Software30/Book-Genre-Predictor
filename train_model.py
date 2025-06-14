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

# Load the preprocessed data
print("Loading preprocessed data...")
try:
    df = pd.read_csv("cleaned_books_genre.csv")
    genre_classes_df = pd.read_csv("genre_classes.csv")
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Make sure 'cleaned_books_genre.csv' and 'genre_classes.csv' exist.")
    exit()

# Split data
print("Splitting data...")
X = df.drop(['genre', 'genre_encoded', 'book_id', 'title', 'authors'], axis=1)
y = df['genre_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define text and numerical features
text_features = ['text_features_combined']
numerical_features = [
    'title_length', 'author_count', 'tag_count',
    'rating_std', 'rating_skew', 'high_rating_ratio'
]

# Create preprocessing pipelines
print("Creating preprocessing pipelines...")
text_transformer = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)))
])

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'text_features_combined'),
        ('num', numerical_transformer, numerical_features)
    ]
)

# Create and train the model
print("\nTraining RandomForestClassifier model...")
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# Evaluate the model
print("\nModel Performance:")
y_pred = model.predict(X_test)
print("\nAccuracy:", (y_pred == y_test).mean())
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
print("\nSaving trained model...")
joblib.dump(model, 'genre_prediction_model.joblib')

print("✅ Model saved successfully.") 