"""
Book Genre Predictor - GUI Application

This module provides a graphical user interface for the Book Genre Predictor application.
It allows users to input book details and get genre predictions with confidence scores.

Key features:
- Clean and modern user interface
- Real-time genre prediction
- Confidence score display
- Error handling and user feedback
"""

import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np
from preprocess import clean_text, get_main_author
import logging
from pathlib import Path
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BookGenrePredictorGUI:
    """Main GUI class for the Book Genre Predictor application."""
    
    def __init__(self, root):
        """
        Initialize the GUI application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Book Genre Predictor")
        self.root.geometry("800x600")
        
        # Load model and resources
        self.load_resources()
        
        # Create GUI elements
        self.create_widgets()
        
        # Configure grid weights
        self.configure_grid()
        
        # Center window
        self.center_window()
        
        # Set theme colors
        self.set_theme()
        
    def load_resources(self):
        """Load the trained model, vectorizer, and genre classes."""
        try:
            # Load model and vectorizer
            self.model = joblib.load('book_genre_model.joblib')
            self.vectorizer = joblib.load('tfidf_vectorizer.joblib')
            
            # Load genre classes
            self.genre_classes = pd.read_csv('genre_classes.csv')
            
            logger.info("Resources loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")
            messagebox.showerror(
                "Error",
                "Failed to load model resources. Please ensure all required files are present."
            )
            self.root.destroy()
    
    def create_widgets(self):
        """Create and configure GUI widgets."""
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Title
        title_label = ttk.Label(
            self.main_frame,
            text="Book Genre Predictor",
            font=("Helvetica", 24, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Input fields
        self.create_input_fields()
        
        # Predict button
        self.predict_button = ttk.Button(
            self.main_frame,
            text="Predict Genre",
            command=self.predict_genre,
            style="Accent.TButton"
        )
        self.predict_button.grid(row=4, column=0, columnspan=2, pady=20)
        
        # Results frame
        self.create_results_frame()
        
        # Loading indicator
        self.loading_label = ttk.Label(
            self.main_frame,
            text="",
            font=("Helvetica", 12)
        )
        self.loading_label.grid(row=6, column=0, columnspan=2, pady=10)
    
    def create_input_fields(self):
        """Create input fields for book details."""
        # Title
        ttk.Label(
            self.main_frame,
            text="Book Title:",
            font=("Helvetica", 12)
        ).grid(row=1, column=0, sticky="w", pady=5)
        
        self.title_entry = ttk.Entry(
            self.main_frame,
            width=40,
            font=("Helvetica", 12)
        )
        self.title_entry.grid(row=1, column=1, sticky="ew", pady=5)
        
        # Author
        ttk.Label(
            self.main_frame,
            text="Author:",
            font=("Helvetica", 12)
        ).grid(row=2, column=0, sticky="w", pady=5)
        
        self.author_entry = ttk.Entry(
            self.main_frame,
            width=40,
            font=("Helvetica", 12)
        )
        self.author_entry.grid(row=2, column=1, sticky="ew", pady=5)
        
        # Keywords
        ttk.Label(
            self.main_frame,
            text="Keywords (comma-separated):",
            font=("Helvetica", 12)
        ).grid(row=3, column=0, sticky="w", pady=5)
        
        self.keywords_entry = ttk.Entry(
            self.main_frame,
            width=40,
            font=("Helvetica", 12)
        )
        self.keywords_entry.grid(row=3, column=1, sticky="ew", pady=5)
    
    def create_results_frame(self):
        """Create frame for displaying prediction results."""
        self.results_frame = ttk.LabelFrame(
            self.main_frame,
            text="Prediction Results",
            padding="10"
        )
        self.results_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
        
        # Predicted genre
        self.genre_label = ttk.Label(
            self.results_frame,
            text="Predicted Genre: ",
            font=("Helvetica", 12)
        )
        self.genre_label.grid(row=0, column=0, sticky="w", pady=5)
        
        # Confidence score
        self.confidence_label = ttk.Label(
            self.results_frame,
            text="Confidence: ",
            font=("Helvetica", 12)
        )
        self.confidence_label.grid(row=1, column=0, sticky="w", pady=5)
    
    def configure_grid(self):
        """Configure grid weights for responsive layout."""
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
    
    def center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def set_theme(self):
        """Set the application theme and styles."""
        style = ttk.Style()
        
        # Configure colors
        style.configure(
            "TFrame",
            background="#f0f0f0"
        )
        style.configure(
            "TLabel",
            background="#f0f0f0",
            foreground="#333333"
        )
        style.configure(
            "TButton",
            font=("Helvetica", 12),
            padding=10
        )
        style.configure(
            "Accent.TButton",
            background="#4a90e2",
            foreground="white"
        )
        style.configure(
            "TLabelframe",
            background="#f0f0f0"
        )
        style.configure(
            "TLabelframe.Label",
            background="#f0f0f0",
            foreground="#333333",
            font=("Helvetica", 12, "bold")
        )
    
    def show_loading(self, show=True):
        """
        Show or hide the loading indicator.
        
        Args:
            show (bool): Whether to show the loading indicator
        """
        if show:
            self.loading_label.config(text="Predicting...")
            self.predict_button.config(state="disabled")
        else:
            self.loading_label.config(text="")
            self.predict_button.config(state="normal")
    
    def predict_genre(self):
        """Handle genre prediction when the predict button is clicked."""
        # Get input values
        title = self.title_entry.get().strip()
        author = self.author_entry.get().strip()
        keywords = self.keywords_entry.get().strip()
        
        # Validate input
        if not title or not author:
            messagebox.showwarning(
                "Input Error",
                "Please enter both title and author."
            )
            return
        
        # Show loading indicator
        self.show_loading(True)
        
        # Run prediction in a separate thread
        threading.Thread(target=self._run_prediction, args=(title, author, keywords)).start()
    
    def _run_prediction(self, title, author, keywords):
        """
        Run the genre prediction in a background thread.
        
        Args:
            title (str): Book title
            author (str): Book author
            keywords (str): Comma-separated keywords
        """
        try:
            # Clean and prepare input
            clean_title = clean_text(title)
            clean_author = clean_text(author)
            clean_keywords = clean_text(keywords)
            
            # Combine features
            text_features = f"{clean_title} {clean_author} {clean_keywords}"
            
            # Transform text features
            X = self.vectorizer.transform([text_features])
            
            # Get prediction and probabilities
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Get confidence score
            confidence = probabilities.max() * 100
            
            # Update results in the main thread
            self.root.after(0, lambda: self._update_results(prediction, confidence))
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror(
                "Prediction Error",
                "An error occurred during prediction. Please try again."
            ))
        
        finally:
            # Hide loading indicator
            self.root.after(0, lambda: self.show_loading(False))
    
    def _update_results(self, prediction, confidence):
        """
        Update the results display with prediction and confidence.
        
        Args:
            prediction (str): Predicted genre
            confidence (float): Confidence score
        """
        self.genre_label.config(
            text=f"Predicted Genre: {prediction}",
            foreground="#4a90e2"
        )
        self.confidence_label.config(
            text=f"Confidence: {confidence:.1f}%",
            foreground="#4a90e2"
        )

def main():
    """Main function to run the GUI application."""
    try:
        root = tk.Tk()
        app = BookGenrePredictorGUI(root)
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        messagebox.showerror(
            "Application Error",
            "An error occurred while starting the application."
        )

if __name__ == "__main__":
    main() 