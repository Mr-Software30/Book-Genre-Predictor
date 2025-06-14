import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import re
from datetime import datetime

# --- Helper functions from preprocess.py (for consistent cleaning) ---
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text).strip().lower()
        return re.sub(r'\s+', ' ', text)
    return ''

def get_main_author(authors):
    if not isinstance(authors, str):
        return ''
    return authors.split(',')[0].strip()

def extract_year(date_str):
    if pd.isna(date_str):
        return None
    try:
        if isinstance(date_str, (int, float)):
            return int(date_str)
        date_obj = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(date_obj):
            return date_obj.year
        year_match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
        if year_match:
            return int(year_match.group())
        return None
    except:
        return None

# --- GUI Class ---
class GenrePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Book Genre Predictor")
        self.root.geometry("800x600")
        self.root.configure(bg='#2C3E50')

        # Load the trained model and genre classes
        try:
            self.model = joblib.load('genre_prediction_model.joblib')
            self.genre_classes_df = pd.read_csv('genre_classes.csv')
            self.genre_id_to_name = dict(zip(self.genre_classes_df['genre_id'], self.genre_classes_df['genre_name']))
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Model or data files not found: {e}. Please run preprocess.py and train_model.py first.")
            root.destroy()
            return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load resources: {e}")
            root.destroy()
            return

        # Styling
        self.style = ttk.Style()
        self.style.configure('Modern.TFrame', background='#2C3E50')
        self.style.configure('Modern.TLabel', background='#2C3E50', foreground='white', font=('Helvetica', 12))
        self.style.configure('Modern.TButton', background='#3498DB', foreground='white', font=('Helvetica', 12, 'bold'))
        self.style.map('Modern.TButton', background=[('active', '#217DBB')])

        # Main Frame
        main_frame = ttk.Frame(root, style='Modern.TFrame', padding="30")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="📚 Book Genre Predictor 📚",
            font=('Helvetica', 20, 'bold'),
            style='Modern.TLabel'
        )
        title_label.pack(pady=15)

        # Instructions
        instructions_label = ttk.Label(
            main_frame,
            text="Enter details about a book, and I'll predict its main genre!",
            style='Modern.TLabel'
        )
        instructions_label.pack(pady=10)

        # Input Fields
        input_grid = ttk.Frame(main_frame, style='Modern.TFrame')
        input_grid.pack(pady=20)

        # Text Features
        ttk.Label(input_grid, text="Book Title:", style='Modern.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        self.title_entry = ttk.Entry(input_grid, width=40, font=('Helvetica', 11))
        self.title_entry.grid(row=0, column=1, pady=5, padx=5)

        ttk.Label(input_grid, text="Author:", style='Modern.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        self.author_entry = ttk.Entry(input_grid, width=40, font=('Helvetica', 11))
        self.author_entry.grid(row=1, column=1, pady=5, padx=5)

        ttk.Label(input_grid, text="Keywords/Tags:", style='Modern.TLabel').grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        self.tags_entry = ttk.Entry(input_grid, width=40, font=('Helvetica', 11))
        self.tags_entry.grid(row=2, column=1, pady=5, padx=5)
        ttk.Label(input_grid, text="(e.g., magic, detective, romance)", font=('Helvetica', 9, 'italic'), foreground='#BDC3C7', background='#2C3E50').grid(row=3, column=1, sticky=tk.W, padx=5)

        # Predict Button
        predict_button = ttk.Button(
            main_frame,
            text="🔮 Predict Genre",
            command=self.predict_genre,
            style='Modern.TButton'
        )
        predict_button.pack(pady=20)

        # Prediction Result Label
        self.result_label = ttk.Label(
            main_frame,
            text="",
            font=('Helvetica', 16, 'bold'),
            style='Modern.TLabel',
            foreground='#27AE60'
        )
        self.result_label.pack(pady=10)

    def predict_genre(self):
        # Get text features
        title = self.title_entry.get()
        author = self.author_entry.get()
        tags = self.tags_entry.get()

        if not title and not tags:
            messagebox.showwarning("Input Error", "Please enter at least a Book Title or some Keywords/Tags.")
            return

        # Preprocess text features
        clean_title_input = clean_text(title)
        main_author_input = get_main_author(author)
        clean_author_input = clean_text(main_author_input)
        clean_tags_input = clean_text(tags)

        # Combine text features
        text_features_combined = f"{clean_title_input} {clean_author_input} {clean_tags_input}"
        
        if not text_features_combined.strip():
            messagebox.showwarning("Input Error", "Combined input is empty. Please provide more details.")
            return

        # Create feature dictionary
        features = {
            'text_features_combined': text_features_combined,
            'title_length': len(clean_title_input),
            'author_count': len(author.split(',')) if author else 1,
            'tag_count': len(tags.split()) if tags else 0,
            'rating_std': 0,  # Default values for rating statistics
            'rating_skew': 0,
            'high_rating_ratio': 0
        }

        # Create DataFrame for prediction
        input_df = pd.DataFrame([features])

        try:
            # Make prediction
            predicted_genre_encoded = self.model.predict(input_df)[0]
            predicted_genre_name = self.genre_id_to_name.get(predicted_genre_encoded, "Unknown")

            # Display result
            self.result_label.config(text=f"Predicted Genre: {predicted_genre_name}", foreground='#27AE60')
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to make prediction: {str(e)}")

# --- Main application entry point ---
def main():
    root = tk.Tk()
    app = GenrePredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 