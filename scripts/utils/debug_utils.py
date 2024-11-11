import sys
import traceback
import logging
import inspect
import os
from functools import wraps
from datetime import datetime

def function_debug_decorator(func):
    """Decorator to add detailed function entry/exit logging"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        func_name = func.__name__
        caller_frame = inspect.currentframe().f_back
        caller_info = inspect.getframeinfo(caller_frame)
        
        # Log function entry
        logger.debug(f"ENTER {func_name} from {caller_info.filename}:{caller_info.lineno}")
        logger.debug(f"Args: {args}, Kwargs: {kwargs}")
        
        try:
            result = func(*args, **kwargs)
            # Log function exit
            logger.debug(f"EXIT {func_name} - Success")
            return result
        except Exception as e:
            # Log error details
            logger.error(f"ERROR in {func_name}: {str(e)}")
            logger.error(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            raise
    return wrapper

class DebugContext:
    """Context manager for debugging code blocks"""
    def __init__(self, name, logger=None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.debug(f"Completed {self.name} in {duration}")
        else:
            self.logger.error(f"Failed {self.name} after {duration}")
            self.logger.error(f"Error: {exc_type.__name__}: {exc_val}")
            self.logger.error(f"Traceback:\n{''.join(traceback.format_tb(exc_tb))}")
        return False  # Don't suppress exceptions

def log_error_details(logger, error, context=""):
    """Detailed error logging function"""
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    # Get the full stack trace
    stack_trace = ''.join(traceback.format_tb(exc_traceback))
    
    # Get local variables from the frame where the error occurred
    local_vars = {}
    try:
        frame = inspect.trace()[-1][0]
        local_vars = frame.f_locals
    except Exception:
        local_vars = {"Note": "Could not capture local variables"}
    
    # Log comprehensive error information
    logger.error(f"Error in {context}: {str(error)}")
    logger.error(f"Error Type: {exc_type.__name__}")
    logger.error(f"Stack Trace:\n{stack_trace}")
    logger.error(f"Local Variables at Error:\n{local_vars}")

def setup_debug_logging(log_dir='debug_logs'):
    """Setup detailed debug logging"""
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    debug_log_path = os.path.join(log_dir, f'debug_{timestamp}.log')
    
    # Configure debug file handler
    debug_handler = logging.FileHandler(debug_log_path)
    debug_handler.setLevel(logging.DEBUG)
    
    # Create detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s\n'
        'Message: %(message)s\n'
    )
    debug_handler.setFormatter(formatter)
    
    # Add handler to root logger
    logging.getLogger().addHandler(debug_handler)
    
    return debug_log_path 