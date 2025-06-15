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
        self.root.geometry("900x700")
        
        # Set theme colors
        self.set_theme()
        
        # Load model and resources
        self.load_resources()
        
        # Create GUI elements
        self.create_widgets()
        
        # Configure grid weights
        self.configure_grid()
        
        # Center window
        self.center_window()
        
    def load_resources(self):
        """Load the trained model, vectorizer, and genre classes."""
        try:
            # Load model and vectorizer
            self.model = joblib.load('book_genre_model.joblib')
            self.vectorizer = joblib.load('tfidf_vectorizer.joblib')
            
            # Load genre classes
            self.genre_classes = pd.read_csv('genre_classes.csv', dtype={'genre_id': int, 'genre_name': str})
            
            logger.info("Resources loaded successfully")
            
            # Debugging: Print genre_classes dtypes
            logger.info(f"genre_classes DataFrame dtypes:\n{self.genre_classes.dtypes}")
            
        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")
            messagebox.showerror(
                "Error",
                "Failed to load model resources. Please ensure all required files are present."
            )
            self.root.destroy()
    
    def create_widgets(self):
        """Create and configure GUI widgets."""
        # Create main frame with padding and background
        self.main_frame = ttk.Frame(self.root, padding="30", style="Main.TFrame")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Title with decorative line
        title_frame = ttk.Frame(self.main_frame, style="Main.TFrame")
        title_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 30))
        
        title_label = ttk.Label(
            title_frame,
            text="Book Genre Predictor",
            font=("Helvetica", 28, "bold"),
            style="Title.TLabel"
        )
        title_label.pack(pady=(0, 10))
        
        # Decorative line
        ttk.Separator(title_frame, orient="horizontal").pack(fill="x", padx=50)
        
        # Input fields
        self.create_input_fields()
        
        # Predict button
        self.predict_button = ttk.Button(
            self.main_frame,
            text="Predict Genre",
            command=self.predict_genre,
            style="Accent.TButton"
        )
        self.predict_button.grid(row=4, column=0, columnspan=2, pady=30)
        
        # Results frame
        self.create_results_frame()
        
        # Loading indicator
        self.loading_label = ttk.Label(
            self.main_frame,
            text="",
            font=("Helvetica", 12),
            style="Info.TLabel"
        )
        self.loading_label.grid(row=6, column=0, columnspan=2, pady=10)
    
    def create_input_fields(self):
        """Create input fields for book details."""
        # Title
        ttk.Label(
            self.main_frame,
            text="Book Title",
            font=("Helvetica", 12, "bold"),
            style="Field.TLabel"
        ).grid(row=1, column=0, sticky="w", pady=(0, 5))
        
        self.title_entry = ttk.Entry(
            self.main_frame,
            width=50,
            font=("Helvetica", 12),
            style="Custom.TEntry"
        )
        self.title_entry.grid(row=1, column=1, sticky="ew", pady=(0, 15))
        
        # Author
        ttk.Label(
            self.main_frame,
            text="Author",
            font=("Helvetica", 12, "bold"),
            style="Field.TLabel"
        ).grid(row=2, column=0, sticky="w", pady=(0, 5))
        
        self.author_entry = ttk.Entry(
            self.main_frame,
            width=50,
            font=("Helvetica", 12),
            style="Custom.TEntry"
        )
        self.author_entry.grid(row=2, column=1, sticky="ew", pady=(0, 15))
        
        # Keywords
        ttk.Label(
            self.main_frame,
            text="Keywords (comma-separated)",
            font=("Helvetica", 12, "bold"),
            style="Field.TLabel"
        ).grid(row=3, column=0, sticky="w", pady=(0, 5))
        
        self.keywords_entry = ttk.Entry(
            self.main_frame,
            width=50,
            font=("Helvetica", 12),
            style="Custom.TEntry"
        )
        self.keywords_entry.grid(row=3, column=1, sticky="ew", pady=(0, 15))
    
    def create_results_frame(self):
        """Create frame for displaying prediction results."""
        self.results_frame = ttk.LabelFrame(
            self.main_frame,
            text="Prediction Results",
            padding="20",
            style="Results.TLabelframe"
        )
        self.results_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
        
        # Predicted genre
        self.genre_label = ttk.Label(
            self.results_frame,
            text="Predicted Genre: ",
            font=("Helvetica", 14),
            style="Result.TLabel"
        )
        self.genre_label.grid(row=0, column=0, sticky="w", pady=5)
        
        # Confidence score
        self.confidence_label = ttk.Label(
            self.results_frame,
            text="Confidence: ",
            font=("Helvetica", 14),
            style="Result.TLabel"
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
        
        # Configure colors for dark mode
        style.configure(
            "Main.TFrame",
            background="#1e1e1e"  # Dark background
        )
        style.configure(
            "Title.TLabel",
            background="#1e1e1e",
            foreground="#ffffff",  # White text
            font=("Helvetica", 28, "bold")
        )
        style.configure(
            "Field.TLabel",
            background="#1e1e1e",
            foreground="#ffffff"  # White text
        )
        style.configure(
            "Info.TLabel",
            background="#1e1e1e",
            foreground="#8e8e8e"  # Muted gray for info text
        )
        style.configure(
            "Result.TLabel",
            background="#1e1e1e",
            foreground="#ffffff"  # White text
        )
        style.configure(
            "Custom.TEntry",
            fieldbackground="#2d2d2d",  # Slightly lighter than background
            foreground="#ffffff",  # White text
            borderwidth=2,
            relief="solid"
        )
        style.configure(
            "TButton",
            font=("Helvetica", 12),
            padding=10
        )
        style.configure(
            "Accent.TButton",
            background="#0a84ff",  # macOS blue
            foreground="white"
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#0071e3"), ("disabled", "#404040")],  # Darker blue on hover
            foreground=[("disabled", "#666666")]
        )
        style.configure(
            "Results.TLabelframe",
            background="#1e1e1e",
            borderwidth=2
        )
        style.configure(
            "Results.TLabelframe.Label",
            background="#1e1e1e",
            foreground="#ffffff",
            font=("Helvetica", 12, "bold")
        )
        
        # Configure the root window
        self.root.configure(bg="#1e1e1e")
        
        # Configure separator color
        style.configure("TSeparator", background="#404040")  # Dark gray separator
    
    def show_loading(self, show=True):
        """
        Show or hide the loading indicator.
        
        Args:
            show (bool): Whether to show the loading indicator
        """
        if show:
            self.loading_label.config(text="Predicting...", foreground="#0a84ff")  # macOS blue
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
            X_text = self.vectorizer.transform([text_features])
            
            # Create numerical features with default values
            # The model expects 9 numerical features: 
            # [rating_std, rating_skew, high_rating_ratio, popularity, rating_score, engagement_score,
            # title_length, author_count, tag_count]
            X_numerical = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
            
            # Combine features
            X = np.hstack([X_text.toarray(), X_numerical])
            
            # Get prediction and probabilities
            prediction = int(self.model.predict(X)[0])
            probabilities = self.model.predict_proba(X)[0]
            
            # Get confidence score
            confidence = probabilities.max() * 100
            
            # Debugging: Print prediction and genre_classes IDs
            logger.info(f"Predicted ID: {prediction} (type: {type(prediction)})")
            logger.info(f"genre_classes['genre_id'] unique values: {self.genre_classes['genre_id'].unique()} (type: {self.genre_classes['genre_id'].dtype})")
            
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
            prediction (int): Predicted genre ID
            confidence (float): Confidence score
        """
        # Debugging: Print prediction and genre_classes IDs
        logger.info(f"Predicted ID: {prediction} (type: {type(prediction)})")
        logger.info(f"genre_classes['genre_id'] unique values: {self.genre_classes['genre_id'].unique()} (type: {self.genre_classes['genre_id'].dtype})")

        # Get genre name from the loaded genre_classes DataFrame
        genre_row = self.genre_classes[self.genre_classes['genre_id'].astype(int) == int(prediction)]
        if not genre_row.empty:
            genre_name = genre_row['genre_name'].values[0]
        else:
            genre_name = "Unknown Genre" # Fallback in case ID is not found

        self.genre_label.config(
            text=f"Predicted Genre: {genre_name}",
            foreground="#30d158"  # macOS green
        )
        self.confidence_label.config(
            text=f"Confidence: {confidence:.1f}%",
            foreground="#30d158"  # macOS green
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