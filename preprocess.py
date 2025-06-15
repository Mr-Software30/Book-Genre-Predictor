"""
Book Genre Predictor - Data Preprocessing Module

This module handles the preprocessing of book data for genre prediction.
It performs text cleaning, feature engineering, and prepares the data for model training.

Key functions:
- clean_text: Cleans and normalizes text data
- get_main_author: Extracts the primary author from a list of authors
- calculate_rating_stats: Computes rating distribution statistics
- infer_main_genre: Determines the main genre from book tags
- preprocess_data_pipeline: Main preprocessing pipeline for the dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define genre mapping for broader genre categories
GENRE_MAPPING = {
    'fiction': ['fiction', 'novel', 'contemporary', 'literary-fiction', 'modern-classics'],
    'fantasy': ['fantasy', 'magic', 'dragons', 'epic-fantasy', 'high-fantasy', 'urban-fantasy', 'paranormal'],
    'mystery': ['mystery', 'thriller', 'crime', 'suspense', 'detective'],
    'romance': ['romance', 'love', 'chick-lit'],
    'young-adult': ['young-adult', 'ya', 'teen', 'childrens'],
    'science-fiction': ['science-fiction', 'sci-fi', 'dystopian', 'space'],
    'historical-fiction': ['historical-fiction', 'historical', 'history'],
    'classics': ['classics', 'classic', 'literature', '19th-century', '20th-century'],
    'biography': ['biography', 'memoir', 'autobiography', 'diary', 'personal-account', 'non-fiction-biography', 'historical-biography', 'war-memoir', 'holocaust', 'world-war-2'],
    'non-fiction': ['non-fiction', 'history', 'science', 'politics', 'philosophy', 'self-help', 'business'],
    'horror': ['horror', 'supernatural', 'gothic'],
    'poetry': ['poetry'],
    'humor': ['humor', 'comedy'],
    'graphic-novel': ['graphic-novels', 'comics', 'manga']
}

# Create inverted mapping for quick lookup
TAG_TO_GENRE = {}
for genre, tags in GENRE_MAPPING.items():
    for tag in tags:
        TAG_TO_GENRE[tag] = genre

def clean_text(text):
    """
    Clean and normalize text data.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned and normalized text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase and remove special characters
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

def get_main_author(authors):
    """
    Extract the primary author from a list of authors.
    
    Args:
        authors (str): Comma-separated list of authors
        
    Returns:
        str: Primary author name
    """
    if not isinstance(authors, str):
        return ''
    return authors.split(',')[0].strip()

def extract_year(date_str):
    """
    Extract year from various date formats.
    
    Args:
        date_str: Date string or number
        
    Returns:
        int or None: Extracted year or None if not found
    """
    if pd.isna(date_str):
        return None
    try:
        # Try to extract year from various date formats
        if isinstance(date_str, (int, float)):
            return int(date_str)
        # Try to parse the date string
        date_obj = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(date_obj):
            return date_obj.year
        # If that fails, try to extract year using regex
        year_match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
        if year_match:
            return int(year_match.group())
        return None
    except:
        return None

def calculate_rating_stats(row):
    """
    Calculate rating distribution statistics.
    
    Args:
        row (pd.Series): Row containing rating counts
        
    Returns:
        pd.Series: Statistics including standard deviation, skewness, and high rating ratio
    """
    # Calculate rating distribution statistics
    total_ratings = sum(row[f'ratings_{i}'] for i in range(1, 6))
    if total_ratings == 0:
        return pd.Series({
            'rating_std': 0,
            'rating_skew': 0,
            'high_rating_ratio': 0
        })
    
    ratings = []
    for i in range(1, 6):
        ratings.extend([i] * row[f'ratings_{i}'])
    
    ratings = np.array(ratings)
    return pd.Series({
        'rating_std': np.std(ratings),
        'rating_skew': pd.Series(ratings).skew(),
        'high_rating_ratio': (row['ratings_4'] + row['ratings_5']) / total_ratings
    })

def infer_main_genre(tags_str):
    """
    Infer the main genre from book tags.
    
    Args:
        tags_str (str): Comma-separated book tags
        
    Returns:
        str: Main genre of the book
    """
    if not isinstance(tags_str, str):
        return 'Other'
    
    tags = tags_str.split()
    genre_counts = {}
    
    # Special handling for biographical works
    biographical_keywords = ['diary', 'memoir', 'autobiography', 'biography', 'personal-account']
    if any(keyword in tags for keyword in biographical_keywords):
        return 'biography'
    
    for tag in tags:
        broad_genre = TAG_TO_GENRE.get(tag, 'Other')
        # Give extra weight to biographical tags
        if broad_genre == 'biography':
            genre_counts[broad_genre] = genre_counts.get(broad_genre, 0) + 2
        else:
            genre_counts[broad_genre] = genre_counts.get(broad_genre, 0) + 1
    
    # Exclude 'Other' if more specific genres are present
    if 'Other' in genre_counts and len(genre_counts) > 1:
        del genre_counts['Other']

    if genre_counts:
        return max(genre_counts, key=genre_counts.get)
    return 'Other'

def preprocess_data_pipeline(input_books_file='archive/books.csv', 
                           input_tags_file='archive/tags.csv', 
                           input_book_tags_file='archive/book_tags.csv',
                           output_cleaned_books_file='cleaned_books_genre.csv',
                           output_genre_classes_file='genre_classes.csv'):
    """
    Main preprocessing pipeline for the dataset.
    
    Args:
        input_books_file (str): Path to books CSV file
        input_tags_file (str): Path to tags CSV file
        input_book_tags_file (str): Path to book-tags CSV file
        output_cleaned_books_file (str): Path to save cleaned books data
        output_genre_classes_file (str): Path to save genre classes
    
    Returns:
        tuple: (cleaned_books_df, genre_classes_df) or (None, None) if error
    """
    try:
        # Load the datasets
        logger.info("Loading datasets...")
        books_df = pd.read_csv(input_books_file)
        tags_df = pd.read_csv(input_tags_file)
        book_tags_df = pd.read_csv(input_book_tags_file)
        
        # Merge tags with books
        logger.info("Processing tags and inferring genres...")
        book_tags = book_tags_df.merge(tags_df, on='tag_id')
        
        # Group by book and get top tags
        book_tags_grouped = book_tags.groupby('goodreads_book_id')['tag_name'].apply(
            lambda x: ' '.join(x.value_counts().nlargest(10).index)
        ).reset_index(name='top_tags_string')
        
        books_df = books_df.merge(book_tags_grouped, 
                                left_on='book_id', 
                                right_on='goodreads_book_id', 
                                how='left')
        books_df['top_tags_string'] = books_df['top_tags_string'].fillna('')
        
        # Infer the main genre
        books_df['genre'] = books_df['top_tags_string'].apply(infer_main_genre)
        
        # Clean and prepare text features
        logger.info("Cleaning text features...")
        books_df['clean_title'] = books_df['title'].apply(clean_text)
        books_df['clean_authors'] = books_df['authors'].apply(get_main_author).apply(clean_text)
        books_df['clean_top_tags'] = books_df['top_tags_string'].apply(clean_text)
        
        # Calculate rating statistics
        logger.info("Calculating rating statistics...")
        rating_stats = books_df.apply(calculate_rating_stats, axis=1)
        books_df = pd.concat([books_df, rating_stats], axis=1)
        
        # Create additional numerical features
        logger.info("Creating additional numerical features...")
        books_df['popularity'] = books_df['ratings_count'].fillna(0)
        books_df['rating_score'] = books_df['average_rating'].fillna(0)
        books_df['engagement_score'] = (books_df['ratings_count'] * books_df['average_rating']).fillna(0)
        
        # Save processed data
        logger.info("Saving processed data...")
        books_df.to_csv(output_cleaned_books_file, index=False)
        
        # Create and save genre classes
        genre_classes = pd.DataFrame({
            'genre_id': range(len(books_df['genre'].unique())),
            'genre_name': sorted(books_df['genre'].unique())
        })
        genre_classes.to_csv(output_genre_classes_file, index=False)
        
        logger.info("Preprocessing completed successfully!")
        return books_df, genre_classes
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Downloading required NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
    
    # Run preprocessing pipeline
    preprocess_data_pipeline() 