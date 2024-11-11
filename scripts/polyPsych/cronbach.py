import pandas as pd
import numpy as np
import logging
import scipy.stats as stats
from typing import Dict, Union, Optional, List, Tuple
from dataclasses import dataclass

# Set up logging with more detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.shape[0] < min_rows:
        raise ValueError(f"Data must have at least {min_rows} rows (participants)")
        
    if df.shape[1] < min_cols:
        raise ValueError(f"Data must have at least {min_cols} columns (items)")
        
    if df.isna().any().any():
        logging.warning("Data contains missing values which may affect results")
        
    if not all(np.issubdtype(dtype, np.number) for dtype in df.dtypes):
        raise TypeError("All columns must contain numeric data")
        
    # Check for constant columns
    if (df.std() == 0).any():
        logging.warning("One or more items have zero variance")
        
    # Check for reasonable value ranges
    if (df < 0).any().any():
        logging.warning("Data contains negative values - verify this is intended")

def calculate_confidence_interval(
    alpha: float, 
    n_items: int, 
    n_subjects: int, 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval for Cronbach's alpha."""
    # Using Fisher's transformation
    z = np.arctanh(alpha)
    se = np.sqrt(2 * (1 - alpha**2) / ((n_items - 1) * (n_subjects - 2)))
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    
    lower = np.tanh(z - z_crit * se)
    upper = np.tanh(z + z_crit * se)
    
    return (lower, upper)

def calculate_item_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive item-level statistics."""
    stats_df = pd.DataFrame({
        'Mean': df.mean(),
        'Std': df.std(),
        'Skewness': df.skew(),
        'Kurtosis': df.kurtosis(),
        'Min': df.min(),
        'Max': df.max()
    })
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
    try:
        validate_data(df)
        logging.info("Data validation passed. Processing comprehensive analysis...")
        
        # Handle missing values
        if handle_missing == 'listwise':
            df = df.dropna()
        elif handle_missing == 'impute':
            df = df.fillna(df.mean())
            
        # Number of items and participants
        N = df.shape[1]
        n_subjects = df.shape[0]
        
        # Variance calculations
        item_variances = df.var(axis=0, ddof=1)
        total_scores = df.sum(axis=1)
        var_total = total_scores.var(ddof=1)
        
        # Calculate alpha
        alpha = (N / (N - 1)) * (1 - item_variances.sum() / var_total)
        
        # Calculate confidence interval and standard error
        ci = calculate_confidence_interval(alpha, N, n_subjects, confidence)
        std_error = (ci[1] - ci[0]) / (2 * stats.norm.ppf((1 + confidence) / 2))
        
        # Item-total correlations
        item_total_correlations = pd.Series(
            {col: df[col].corr(total_scores - df[col]) for col in df.columns},
            name="Item-Total Correlations"
        )
        
        # Alpha if item deleted
        alpha_if_deleted = pd.Series(
            {col: cronbach_alpha(df.drop(columns=col))['alpha'] for col in df.columns},
            name="Alpha if Item Deleted"
        )
        
        # Calculate item statistics
        item_stats = calculate_item_statistics(df)
        
        # Calculate inter-item correlations
        inter_item_corr = df.corr()
        
        # Calculate scale statistics
        scale_stats = {
            'mean': total_scores.mean(),
            'variance': var_total,
            'std_dev': np.sqrt(var_total),
            'n_items': N,
            'n_subjects': n_subjects
        }
        
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
        
        return results
        
    except Exception as e:
        logging.error(f"Error calculating Cronbach's alpha: {str(e)}")
        raise

def display_results(results: CronbachResults, detailed: bool = True) -> None:
    """Display formatted results of the Cronbach's alpha analysis."""
    print("\nCRONBACH'S ALPHA ANALYSIS")
    print("=" * 70)
    
    print(f"\nOverall Cronbach's Alpha: {results.alpha:.3f}")
    print(f"95% Confidence Interval: ({results.confidence_interval[0]:.3f}, {results.confidence_interval[1]:.3f})")
    print(f"Standard Error: {results.std_error:.3f}")
    
    print("\nScale Statistics:")
    print("-" * 70)
    for key, value in results.scale_statistics.items():
        print(f"{key.replace('_', ' ').title()}: {value:.3f}")
    
    print("\nItem Analysis:")
    print("-" * 70)
    analysis = pd.DataFrame({
        'Item-Total Correlation': results.item_total_correlations,
        'Alpha if Item Deleted': results.alpha_if_deleted
    })
    print(analysis.round(3))
    
    if detailed:
        print("\nDetailed Item Statistics:")
        print("-" * 70)
        print(results.item_statistics.round(3))
        
        print("\nInter-Item Correlations:")
        print("-" * 70)
        print(results.inter_item_correlations.round(3))

# Example usage with error handling
if __name__ == "__main__":
    try:
        # Sample data
        data = {
            'Item1': [4, 3, 5, 2, 4, 5, 3, 2],
            'Item2': [3, 4, 2, 5, 3, 4, 4, 3],
            'Item3': [5, 4, 3, 4, 5, 3, 4, 4],
            'Item4': [2, 3, 4, 3, 4, 4, 3, 5]
        }
        
        df = pd.DataFrame(data)
        logging.info("Processing sample data...")
        
        results = cronbach_alpha(df)
        display_results(results)
        
    except Exception as e:
        logging.error(f"Program execution failed: {str(e)}")
        raise