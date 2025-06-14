# Book Genre Predictor 📚

A machine learning project that predicts book genres based on their titles, authors, and keywords. The project includes a user-friendly GUI for easy interaction.

## Features

- **Genre Prediction**: Predicts book genres using a Random Forest Classifier
- **User-Friendly GUI**: Simple interface for entering book details
- **Content-Based Analysis**: Uses text features and metadata for accurate predictions
- **Multiple Genre Support**: Can predict 14 different genres including:
  - Fiction
  - Fantasy
  - Mystery
  - Romance
  - Science Fiction
  - Classics
  - Biography
  - And more...

## Project Structure

```
Book-Genre-Predictor/
├── archive/                 # Original datasets
│   ├── books.csv           # Book information
│   ├── tags.csv            # Tag definitions
│   └── book_tags.csv       # Book-tag relationships
├── preprocess.py           # Data preprocessing script
├── train_model.py          # Model training script
├── gui_app.py             # GUI application
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mr-Software30/Book-Genre-Predictor.git
cd Book-Genre-Predictor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. First, preprocess the data:
```bash
python preprocess.py
```

2. Train the model:
```bash
python train_model.py
```

3. Launch the GUI:
```bash
python gui_app.py
```

4. In the GUI:
   - Enter the book title
   - Enter the author's name
   - Add relevant keywords/tags
   - Click "Predict Genre" to get the prediction

## Example Predictions

Try these examples:
- "The Lord of the Rings" by J.R.R. Tolkien (Keywords: magic elves rings quest)
- "Pride and Prejudice" by Jane Austen (Keywords: romance marriage 19th century)
- "The Da Vinci Code" by Dan Brown (Keywords: thriller crime art history)

## Technical Details

- **Model**: Random Forest Classifier
- **Features Used**:
  - Text features (title, author, tags)
  - Title length
  - Author count
  - Tag count
  - Rating statistics
- **Accuracy**: ~80% on test data

## Requirements

- Python 3.x
- scikit-learn
- pandas
- numpy
- tkinter
- joblib

## Author

- [Mr-Software30](https://github.com/Mr-Software30)

## License

This project is open source and available under the MIT License.

--- 