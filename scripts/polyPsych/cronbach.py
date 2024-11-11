import pandas as pd
import numpy as np
import logging
import scipy.stats as stats
from typing import Dict, Union, Optional, List, Tuple
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import traceback
import os
from datetime import datetime
# At the top of the file, after imports
import os
from datetime import datetime

def setup_logging():
    """Configure logging with both file and console handlers"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamp for log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'cronbach_{timestamp}.log')

    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler with detailed formatting
    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.DEBUG)
    f_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s'
    )
    f_handler.setFormatter(f_formatter)

    # Console handler with simpler formatting
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_formatter = logging.Formatter('%(levelname)s: %(message)s')
    c_handler.setFormatter(c_formatter)

    # Add handlers to logger
    logger.addHandler(f_handler)
    logger.addHandler(c_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# Set up logging with more detailed configuration
def setup_logging():
    """Configure logging with both file and console handlers"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamp for log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'cronbach_{timestamp}.log')

    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler with detailed formatting
    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.DEBUG)
    f_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s'
    )
    f_handler.setFormatter(f_formatter)

    # Console handler with simpler formatting
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_formatter = logging.Formatter('%(levelname)s: %(message)s')
    c_handler.setFormatter(c_formatter)

    # Add handlers to logger
    logger.addHandler(f_handler)
    logger.addHandler(c_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

logger = setup_logging()

@dataclass
class CronbachResults:
    """Class to store Cronbach's alpha analysis results."""
    alpha: float
    item_total_correlations: pd.Series
    alpha_if_deleted: pd.Series
    confidence_interval: Tuple[float, float]
    std_error: float
    item_statistics: pd.DataFrame
    inter_item_correlations: pd.DataFrame
    scale_statistics: Dict[str, float]

def validate_data(df: pd.DataFrame, min_rows: int = 2, min_cols: int = 2) -> None:
    """
    Validate input data meets requirements for Cronbach's alpha calculation.
    
    Args:
        df: Input DataFrame to validate
        min_rows: Minimum required number of rows (participants)
        min_cols: Minimum required number of columns (items)
        
    Raises:
        TypeError: If input is not a DataFrame or contains non-numeric data
        ValueError: If data doesn't meet size requirements or has other issues
    """
    logger.debug("Starting data validation")
    
    if not isinstance(df, pd.DataFrame):
        logger.error("Input is not a pandas DataFrame")
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.shape[0] < min_rows:
        logger.error(f"Insufficient rows: {df.shape[0]} < {min_rows}")
        raise ValueError(f"Data must have at least {min_rows} rows (participants)")
        
    if df.shape[1] < min_cols:
        logger.error(f"Insufficient columns: {df.shape[1]} < {min_cols}")
        raise ValueError(f"Data must have at least {min_cols} columns (items)")
        
    if df.isna().any().any():
        logger.warning("Data contains missing values which may affect results")
        
    if not all(np.issubdtype(dtype, np.number) for dtype in df.dtypes):
        logger.error("Non-numeric data found in DataFrame")
        raise TypeError("All columns must contain numeric data")
        
    # Check for constant columns
    if (df.std() == 0).any():
        logger.warning("One or more items have zero variance")
        
    # Check for reasonable value ranges
    if (df < 0).any().any():
        logger.warning("Data contains negative values - verify this is intended")
        
    logger.debug("Data validation completed successfully")

def calculate_confidence_interval(
    alpha: float, 
    n_items: int, 
    n_subjects: int, 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval for Cronbach's alpha."""
    logger.debug(f"Calculating confidence interval with alpha={alpha}, n_items={n_items}, n_subjects={n_subjects}")
    
    # Using Fisher's transformation
    if alpha <= -1 or alpha >= 1:
        logger.error("Alpha must be between -1 and 1 for Fisher's transformation.")
        raise ValueError("Alpha must be between -1 and 1 to calculate confidence interval.")
    
    z = np.arctanh(alpha)
    se = np.sqrt(2 * (1 - alpha**2) / ((n_items - 1) * (n_subjects - 2)))
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    
    lower = np.tanh(z - z_crit * se)
    upper = np.tanh(z + z_crit * se)
    
    logger.debug(f"Confidence interval calculated: ({lower:.3f}, {upper:.3f})")
    return (lower, upper)

def calculate_item_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive item-level statistics."""
    logger.debug("Calculating item statistics")
    
    stats_df = pd.DataFrame({
        'Mean': df.mean(),
        'Std': df.std(),
        'Skewness': df.skew(),
        'Kurtosis': df.kurtosis(),
        'Min': df.min(),
        'Max': df.max()
    })
    
    logger.debug("Item statistics calculation completed")
    return stats_df

def cronbach_alpha(
    df: pd.DataFrame, 
    confidence: float = 0.95,
    handle_missing: str = 'pairwise'
) -> CronbachResults:
    """
    Calculate Cronbach's alpha and comprehensive reliability statistics.
    
    Args:
        df: pandas DataFrame with items as columns and participants as rows
        confidence: Confidence level for interval estimation (default: 0.95)
        handle_missing: Method for handling missing values ('pairwise', 'listwise', or 'impute')
        
    Returns:
        CronbachResults object containing:
        - alpha: Cronbach's alpha coefficient
        - item_total_correlations: Correlation of each item with total score
        - alpha_if_deleted: Alpha coefficient if each item is deleted
        - confidence_interval: Confidence interval for alpha
        - std_error: Standard error of alpha
        - item_statistics: Detailed item-level statistics
        - inter_item_correlations: Correlation matrix between items
        - scale_statistics: Overall scale statistics
    """
    logger.info(f"Starting Cronbach's alpha calculation with {df.shape[1]} items and {df.shape[0]} subjects")
    try:
        validate_data(df)
        
        logger.debug(f"Using confidence level: {confidence}")
        logger.debug(f"Missing data handling method: {handle_missing}")
        
        # Handle missing values
        if handle_missing == 'listwise':
            logger.debug("Using listwise deletion for missing values")
            df = df.dropna()
        elif handle_missing == 'impute':
            logger.debug("Imputing missing values with column means")
            df = df.fillna(df.mean())
        elif handle_missing == 'pairwise':
            logger.debug("Using pairwise deletion for missing values")
            # Pairwise is default in pandas correlations
        else:
            logger.error(f"Unknown handle_missing option: {handle_missing}")
            raise ValueError("handle_missing must be 'pairwise', 'listwise', or 'impute'")
            
        # Number of items and participants
        N = df.shape[1]
        n_subjects = df.shape[0]
        logger.debug(f"Processing data with {N} items and {n_subjects} subjects")
        
        if var_total := df.sum(axis=1).var(ddof=1) == 0:
            logger.error("Total variance is zero, cannot compute Cronbach's alpha.")
            raise ValueError("Total variance is zero, cannot compute Cronbach's alpha.")
        
        # Variance calculations
        item_variances = df.var(axis=0, ddof=1)
        total_scores = df.sum(axis=1)
        var_total = total_scores.var(ddof=1)
        logger.debug(f"Total variance: {var_total:.3f}")
        
        if var_total == 0:
            logger.error("Total score variance is zero, cannot compute Cronbach's alpha.")
            raise ValueError("Total score variance is zero, cannot compute Cronbach's alpha.")
        
        # Calculate alpha
        alpha = (N / (N - 1)) * (1 - item_variances.sum() / var_total)
        logger.info(f"Calculated Cronbach's alpha: {alpha:.3f}")
        
        # Calculate confidence interval and standard error
        ci = calculate_confidence_interval(alpha, N, n_subjects, confidence)
        std_error = (ci[1] - ci[0]) / (2 * stats.norm.ppf((1 + confidence) / 2))
        logger.debug(f"Standard error: {std_error:.3f}")
        
        # Item-total correlations
        logger.debug("Calculating item-total correlations")
        item_total_correlations = pd.Series(
            {col: df[col].corr(total_scores - df[col]) for col in df.columns},
            name="Item-Total Correlations"
        )
        
        # Alpha if item deleted
        logger.debug("Calculating alpha if item deleted")
        alpha_if_deleted = pd.Series(
            {col: cronbach_alpha(df.drop(columns=col), confidence, handle_missing).alpha 
             if df.shape[1] > 2  # Only calculate if more than 2 columns remain
             else float('nan')   # Otherwise return NaN
             for col in df.columns},
            name="Alpha if Item Deleted"
        )
        
        # Calculate item statistics
        item_stats = calculate_item_statistics(df)
        
        # Calculate inter-item correlations
        logger.debug("Calculating inter-item correlations")
        inter_item_corr = df.corr()
        
        # Calculate scale statistics
        scale_stats = {
            'mean': total_scores.mean(),
            'variance': var_total,
            'std_dev': np.sqrt(var_total),
            'n_items': N,
            'n_subjects': n_subjects
        }
        logger.debug("Scale statistics calculated")
        
        results = CronbachResults(
            alpha=alpha,
            item_total_correlations=item_total_correlations,
            alpha_if_deleted=alpha_if_deleted,
            confidence_interval=ci,
            std_error=std_error,
            item_statistics=item_stats,
            inter_item_correlations=inter_item_corr,
            scale_statistics=scale_stats
        )
        
        logger.info(f"Calculation complete. Alpha: {alpha:.3f}")
        return results
        
    except Exception as e:
        logger.error("Failed to calculate Cronbach's alpha", exc_info=True)
        raise

def display_results(results: CronbachResults, detailed: bool = True) -> str:
    """Generate formatted results of the Cronbach's alpha analysis."""
    logger.info("Generating analysis results for display")
    
    output = []
    output.append("\nCRONBACH'S ALPHA ANALYSIS")
    output.append("=" * 70)
    
    output.append(f"\nOverall Cronbach's Alpha: {results.alpha:.3f}")
    output.append(f"{int(100*0.95)}% Confidence Interval: ({results.confidence_interval[0]:.3f}, {results.confidence_interval[1]:.3f})")
    output.append(f"Standard Error: {results.std_error:.3f}")
    
    output.append("\nScale Statistics:")
    output.append("-" * 70)
    for key, value in results.scale_statistics.items():
        output.append(f"{key.replace('_', ' ').title()}: {value:.3f}")
    
    output.append("\nItem Analysis:")
    output.append("-" * 70)
    analysis = pd.DataFrame({
        'Item-Total Correlation': results.item_total_correlations,
        'Alpha if Item Deleted': results.alpha_if_deleted
    })
    output.append(analysis.round(3).to_string())
    
    if detailed:
        output.append("\nDetailed Item Statistics:")
        output.append("-" * 70)
        output.append(results.item_statistics.round(3).to_string())
        
        output.append("\nInter-Item Correlations:")
        output.append("-" * 70)
        output.append(results.inter_item_correlations.round(3).to_string())
        
    logger.debug("Results generation completed")
    return "\n".join(output)

class CronbachAlphaApp:
    def __init__(self, root):
        logger.info("Initializing Cronbach's Alpha Calculator GUI")
        self.root = root
        self.root.title("Cronbach's Alpha Calculator")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.df = None
        self.results = None
        self.confidence_level = tk.DoubleVar(value=0.95)
        self.handle_missing = tk.StringVar(value='pairwise')
        
        logger.debug("Creating GUI widgets")
        self.create_widgets()
        logger.info("GUI initialization complete")
    
    def create_widgets(self):
        # Menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load CSV", command=self.load_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        # Frame for settings
        settings_frame = ttk.LabelFrame(self.root, text="Settings")
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Confidence Level
        ttk.Label(settings_frame, text="Confidence Level:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        confidence_entry = ttk.Entry(settings_frame, textvariable=self.confidence_level, width=10)
        confidence_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Handle Missing Data
        ttk.Label(settings_frame, text="Handle Missing Data:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        handle_options = ['pairwise', 'listwise', 'impute']
        handle_menu = ttk.OptionMenu(settings_frame, self.handle_missing, self.handle_missing.get(), *handle_options)
        handle_menu.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Run Button
        run_button = ttk.Button(settings_frame, text="Run Analysis", command=self.run_analysis)
        run_button.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Results Frame
        results_frame = ttk.LabelFrame(self.root, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, wrap=tk.NONE)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar_y = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.configure(yscrollcommand=scrollbar_y.set)
        
        # Export Button
        export_button = ttk.Button(self.root, text="Export Results", command=self.export_results)
        export_button.pack(pady=5)
        
        # Status Bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_status("Ready")
    
    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                logger.debug(f"Loaded data from {file_path}")
                self.update_status(f"Loaded data from {file_path}")
                messagebox.showinfo("Success", f"Data loaded successfully from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load CSV file: {str(e)}", exc_info=True)
                messagebox.showerror("Error", f"Failed to load CSV file:\n{str(e)}")
                self.update_status("Failed to load data")
    
    def run_analysis(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a CSV file first.")
            return
        
        try:
            confidence = self.confidence_level.get()
            if not (0 < confidence < 1):
                raise ValueError("Confidence level must be between 0 and 1.")
            
            handle_missing = self.handle_missing.get()
            logger.debug(f"Running analysis with confidence={confidence}, handle_missing={handle_missing}")
            self.update_status("Running analysis...")
            
            self.results = cronbach_alpha(
                self.df,
                confidence=confidence,
                handle_missing=handle_missing
            )
            result_str = display_results(self.results, detailed=True)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result_str)
            self.update_status("Analysis completed successfully.")
            logger.info("Analysis completed and results displayed.")
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n{str(e)}\n\n{traceback_str}")
            self.update_status("Analysis failed.")
    
    def export_results(self):
        if self.results is None:
            messagebox.showwarning("No Results", "Run the analysis first to export results.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Save Results As"
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(display_results(self.results, detailed=True))
                logger.debug(f"Results exported to {file_path}")
                self.update_status(f"Results exported to {file_path}")
                messagebox.showinfo("Success", f"Results exported successfully to {file_path}")
            except Exception as e:
                logger.error(f"Failed to export results: {str(e)}", exc_info=True)
                messagebox.showerror("Error", f"Failed to export results:\n{str(e)}")
                self.update_status("Failed to export results.")
    
    def show_about(self):
        messagebox.showinfo(
            "About Cronbach's Alpha Calculator",
            "Cronbach's Alpha Calculator\nVersion 1.0\nDeveloped with Tkinter"
        )
    
    def update_status(self, message: str):
        self.status_var.set(message)
        logger.debug(f"Status updated: {message}")

# Example usage with error handling
def main():
    try:
        logger.info("Starting Cronbach's Alpha GUI Application")
        root = tk.Tk()
        app = CronbachAlphaApp(root)
        
        # Set up exception handling for the GUI
        def handle_exception(exc_type, exc_value, exc_traceback):
            logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
            messagebox.showerror("Error", f"An unexpected error occurred:\n{exc_value}")
        
        # Install exception handler
        sys.excepthook = handle_exception
        
        logger.info("Entering main event loop")
        root.mainloop()
        logger.info("Application closed normally")
    except Exception as e:
        logger.critical("Fatal error in main loop", exc_info=True)
        messagebox.showerror("Fatal Error", f"Application failed to start:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()