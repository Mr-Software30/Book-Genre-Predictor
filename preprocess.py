import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime

def clean_text(text):
    if isinstance(text, str):
        # Remove special characters, extra spaces, and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text).strip().lower()
        return re.sub(r'\s+', ' ', text) # Replace multiple spaces with single
    return ''

def get_main_author(authors):
    if not isinstance(authors, str):
        return ''
    return authors.split(',')[0].strip()

def extract_year(date_str):
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

# Load the datasets
print("Loading datasets...")
try:
    books_df = pd.read_csv("archive/books.csv")
    tags_df = pd.read_csv("archive/tags.csv")
    book_tags_df = pd.read_csv("archive/book_tags.csv")
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Make sure 'archive' folder with datasets is in the same directory.")
    exit()

# Merge tags with books
print("Processing tags and inferring genres...")
book_tags = book_tags_df.merge(tags_df, on='tag_id')

# Define a mapping for broader genres from specific tags
# This is crucial for defining our target variable
genre_mapping = {
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

# Invert mapping for quick lookup: tag -> broad_genre
tag_to_broad_genre = {}
for broad_genre, tags in genre_mapping.items():
    for tag in tags:
        tag_to_broad_genre[tag] = broad_genre

# Function to infer the main genre based on top tags
def infer_main_genre(tags_str):
    if not isinstance(tags_str, str):
        return 'Other'
    tags = tags_str.split()
    genre_counts = {}
    
    # Special handling for biographical works
    biographical_keywords = ['diary', 'memoir', 'autobiography', 'biography', 'personal-account']
    if any(keyword in tags for keyword in biographical_keywords):
        return 'biography'
    
    for tag in tags:
        broad_genre = tag_to_broad_genre.get(tag, 'Other')
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

# Group by book and get top tags based on count, then infer genre
book_tags_grouped = book_tags.groupby('goodreads_book_id')['tag_name'].apply(
    lambda x: ' '.join(x.value_counts().nlargest(10).index) # Get top 10 tags by count
).reset_index(name='top_tags_string')

books_df = books_df.merge(book_tags_grouped, left_on='book_id', right_on='goodreads_book_id', how='left')
books_df['top_tags_string'] = books_df['top_tags_string'].fillna('')

# Infer the main genre (our target variable 'genre')
books_df['genre'] = books_df['top_tags_string'].apply(infer_main_genre)

# Clean and prepare text features
print("Cleaning text features...")
books_df['clean_title'] = books_df['title'].apply(clean_text)
books_df['clean_authors'] = books_df['authors'].apply(get_main_author).apply(clean_text)
books_df['clean_top_tags'] = books_df['top_tags_string'].apply(clean_text)

# Calculate rating statistics
print("Calculating rating statistics...")
rating_stats = books_df.apply(calculate_rating_stats, axis=1)
books_df = pd.concat([books_df, rating_stats], axis=1)

# Create additional features
books_df['title_length'] = books_df['clean_title'].str.len()
books_df['author_count'] = books_df['authors'].str.count(',') + 1
books_df['tag_count'] = books_df['top_tags_string'].str.count(' ') + 1

# Combine text features for TF-IDF
books_df['text_features_combined'] = (
    books_df['clean_title'] + ' ' + 
    books_df['clean_authors'] + ' ' + 
    books_df['clean_top_tags']
)

# Select final features and target
feature_columns = [
    'book_id', 'title', 'authors', 'text_features_combined',
    'title_length', 'author_count', 'tag_count',
    'rating_std', 'rating_skew', 'high_rating_ratio',
    'genre'
]

final_df = books_df[feature_columns].copy()

# Filter out books where genre could not be confidently inferred
final_df = final_df[final_df['genre'] != 'Other']

# Encode the genre labels
label_encoder = LabelEncoder()
final_df['genre_encoded'] = label_encoder.fit_transform(final_df['genre'])

# Save the label encoder classes correctly (mapping integer ID to genre name)
genre_classes_df = pd.DataFrame({
    'genre_id': label_encoder.transform(label_encoder.classes_), 
    'genre_name': label_encoder.classes_
})
genre_classes_df.to_csv("genre_classes.csv", index=False)

# Save cleaned data
print("Saving processed data...")
final_df.to_csv("cleaned_books_genre.csv", index=False)

print("\n✅ Preprocessing complete. File saved as cleaned_books_genre.csv")
print(f"Total books processed for genre prediction: {len(final_df)}")
print("\nGenre Distribution:")
print(final_df['genre'].value_counts())
print(f"Number of unique genres: {len(final_df['genre'].unique())}") 