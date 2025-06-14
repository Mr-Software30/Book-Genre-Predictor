"""
Book Genre Predictor - GUI Application

This module provides a graphical user interface for the Book Genre Predictor.
It allows users to input book details and get genre predictions using the trained model.

Key features:
- Input fields for book title, author, and keywords
- Real-time genre prediction
- Display of prediction confidence
- User-friendly interface with clear instructions
"""

import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np
from preprocess import clean_text

class BookGenrePredictorGUI:
    """
    GUI application for book genre prediction.
    
    This class creates and manages the graphical user interface for the
    Book Genre Predictor application.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Book Genre Predictor")
        self.root.geometry("600x500")
        
        # Load the trained model and vectorizer
        try:
            self.model = joblib.load('book_genre_model.joblib')
            self.vectorizer = joblib.load('tfidf_vectorizer.joblib')
            self.genre_classes = pd.read_csv('genre_classes.csv')
        except FileNotFoundError:
            messagebox.showerror("Error", "Model files not found. Please train the model first.")
            root.destroy()
            return
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create and arrange GUI widgets."""
        # Title
        title_label = ttk.Label(
            self.root,
            text="Book Genre Predictor",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=20)
        
        # Input frame
        input_frame = ttk.Frame(self.root)
        input_frame.pack(padx=20, pady=10, fill="x")
        
        # Book title input
        ttk.Label(input_frame, text="Book Title:").pack(anchor="w")
        self.title_entry = ttk.Entry(input_frame, width=50)
        self.title_entry.pack(pady=5)
        
        # Author input
        ttk.Label(input_frame, text="Author:").pack(anchor="w")
        self.author_entry = ttk.Entry(input_frame, width=50)
        self.author_entry.pack(pady=5)
        
        # Keywords input
        ttk.Label(input_frame, text="Keywords/Tags (comma-separated):").pack(anchor="w")
        self.keywords_entry = ttk.Entry(input_frame, width=50)
        self.keywords_entry.pack(pady=5)
        
        # Predict button
        predict_button = ttk.Button(
            self.root,
            text="Predict Genre",
            command=self.predict_genre
        )
        predict_button.pack(pady=20)
        
        # Result frame
        self.result_frame = ttk.Frame(self.root)
        self.result_frame.pack(padx=20, pady=10, fill="x")
        
        # Result labels
        self.genre_label = ttk.Label(
            self.result_frame,
            text="Predicted Genre: ",
            font=("Helvetica", 12)
        )
        self.genre_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(
            self.result_frame,
            text="Confidence: ",
            font=("Helvetica", 12)
        )
        self.confidence_label.pack(pady=5)
    
    def predict_genre(self):
        """
        Predict the genre based on user input.
        
        This method:
        1. Gets input from the user
        2. Preprocesses the input
        3. Makes a prediction using the trained model
        4. Updates the GUI with the results
        """
        # Get input
        title = self.title_entry.get().strip()
        author = self.author_entry.get().strip()
        keywords = self.keywords_entry.get().strip()
        
        if not title or not author:
            messagebox.showwarning("Warning", "Please enter both title and author.")
            return
        
        # Preprocess input
        content_features = clean_text(f"{title} {author} {keywords}")
        
        # Transform text features
        X_text = self.vectorizer.transform([content_features])
        
        # Create numerical features (using default values for now)
        X_numerical = np.array([[0, 0, 0]])  # [popularity, rating_score, engagement_score]
        
        # Combine features
        X = np.hstack([X_text.toarray(), X_numerical])
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = probabilities[prediction]
        
        # Get genre name
        genre_name = self.genre_classes[
            self.genre_classes['genre_encoded'] == prediction
        ]['genre'].values[0]
        
        # Update GUI
        self.genre_label.config(text=f"Predicted Genre: {genre_name}")
        self.confidence_label.config(text=f"Confidence: {confidence:.1%}")

def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = BookGenrePredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 