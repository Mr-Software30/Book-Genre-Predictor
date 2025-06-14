# Book Genre Predictor

This is an end-to-end machine learning project designed to predict the genre of a book based on its title, author, and associated keywords/tags. The project includes data preprocessing, model training, and a graphical user interface (GUI) for interactive predictions.

## Project Structure

```
.
├── archive/
│   ├── books.csv
│   ├── book_tags.csv
│   └── tags.csv
├── cleaned_books_genre.csv
├── genre_classes.csv
├── genre_prediction_model.joblib
├── tfidf_vectorizer.joblib
├── preprocess.py
├── train_model.py
├── gui_app.py
└── requirements.txt
```

## How It Works

1.  **`preprocess.py`**:
    *   Loads raw data from `archive/books.csv`, `archive/tags.csv`, and `archive/book_tags.csv`.
    *   Cleans text data (titles, authors, tags).
    *   Infers a primary genre for each book based on its top tags.
    *   Saves the cleaned and prepared data to `cleaned_books_genre.csv` and the genre mappings to `genre_classes.csv`.

2.  **`train_model.py`**:
    *   Loads the `cleaned_books_genre.csv` and `genre_classes.csv`.
    *   Uses TF-IDF Vectorization to convert text features into numerical representations.
    *   Trains a RandomForestClassifier model to predict book genres.
    *   Evaluates the model's performance.
    *   Saves the trained model (`genre_prediction_model.joblib`) and the TF-IDF vectorizer (`tfidf_vectorizer.joblib`) for use in the GUI.

3.  **`gui_app.py`**:
    *   Provides a graphical user interface built with Tkinter.
    *   Loads the trained model and TF-IDF vectorizer.
    *   Allows users to input a book's title, author, and keywords/tags.
    *   Predicts and displays the likely genre of the input book.

## Setup and Running the Project

Follow these steps to set up and run the Book Genre Predictor:

### 1. **Clone the Repository**
```bash
git clone <your_repo_url>
cd Book-genre-Predictor
```
*(Note: Replace `<your_repo_url>` with your actual GitHub repository URL)*

### 2. **Install Dependencies**
Ensure you have Python 3 installed. Then, install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. **Place Raw Data**
Make sure your `archive/` folder (containing `books.csv`, `tags.csv`, `book_tags.csv`) is present in the project root.

### 4. **Preprocess the Data**
Run the preprocessing script to prepare your dataset. This needs to be run only once, or whenever the raw data changes.
```bash
python3 preprocess.py
```

### 5. **Train the Machine Learning Model**
Run the model training script. This will train the genre prediction model and save it. This also needs to be run only once.
```bash
python3 train_model.py
```

### 6. **Run the GUI Application**
Once the data is preprocessed and the model is trained, you can launch the GUI:
```bash
python3 gui_app.py
```

Now, you can enter book details in the GUI to get genre predictions!

## Clean Code Practices

This project adheres to Clean Code principles, focusing on:
*   **Modularity**: Code is organized into separate, focused files (`preprocess.py`, `train_model.py`, `gui_app.py`).
*   **Well-named variables and functions**: Clear and descriptive naming is used throughout the codebase.
*   **Readability**: Comments are minimal but helpful, explaining complex logic or non-obvious steps.

## Contributing

Feel free to fork this repository, contribute, and improve!

--- 