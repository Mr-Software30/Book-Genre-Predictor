# Book Genre Predictor

A machine learning application that predicts book genres based on title, author, and keywords. This project demonstrates end-to-end machine learning implementation with a user-friendly GUI interface.

## Features

- **Genre Prediction**: Predicts book genres using a trained machine learning model
- **Confidence Scores**: Shows prediction confidence for better decision making
- **Modern GUI**: Clean and intuitive user interface built with Tkinter
- **Real-time Processing**: Fast and responsive predictions
- **Error Handling**: Robust error handling and user feedback

## Project Structure

```
Book-rating-Project-STSE/
├── archive/                    # Dataset files
│   ├── books.csv              # Book information
│   ├── tags.csv               # Book tags
│   └── book_tags.csv          # Book-tag mappings
├── gui_app.py                 # GUI application
├── preprocess.py              # Data preprocessing module
├── train_model.py             # Model training module
├── book_genre_model.joblib    # Trained model
├── tfidf_vectorizer.joblib    # TF-IDF vectorizer
├── genre_classes.csv          # Genre class mappings
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Book-rating-Project-STSE.git
   cd Book-rating-Project-STSE
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK data:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Usage

1. Run the GUI application:
   ```bash
   python gui_app.py
   ```

2. Enter book details:
   - Book Title
   - Author
   - Keywords (optional, comma-separated)

3. Click "Predict Genre" to get the prediction

## Model Training

To train the model with your own data:

1. Place your dataset files in the `archive` directory:
   - `books.csv`: Book information
   - `tags.csv`: Book tags
   - `book_tags.csv`: Book-tag mappings

2. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```

3. Train the model:
   ```bash
   python train_model.py
   ```

## Technical Details

### Data Preprocessing

The preprocessing pipeline includes:
- Text cleaning and normalization
- Feature engineering
- Genre inference from tags
- Rating statistics calculation

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Features**:
  - TF-IDF text features
  - Numerical features (ratings, popularity)
- **Evaluation**: Cross-validation with classification metrics

### GUI Implementation

- Built with Tkinter
- Responsive design
- Asynchronous prediction processing
- Error handling and user feedback

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: [Goodreads Books Dataset](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks)
- Libraries: scikit-learn, pandas, numpy, NLTK

## Contact

For questions or feedback, please open an issue in the repository. 