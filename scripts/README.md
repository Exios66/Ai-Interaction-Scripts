# Open-Ended Response Analysis Tool

## Overview

This tool provides comprehensive analysis of open-ended survey responses, specifically designed for analyzing responses about fake news definitions and verification steps. It includes functionality for coding responses, analyzing themes, and quantifying verification steps.

## Features

### Core Functionality

- Load and analyze CSV files containing survey responses
- Code responses using predefined coding schemes
- Analyze common themes in responses
- Quantify verification steps
- Generate statistical summaries
- Comprehensive error logging and debugging

### Analysis Types

1. **Definition Analysis**
   - Coding of responses based on predefined categories
   - Theme identification and frequency analysis
   - Automated categorization of responses

2. **Verification Steps Analysis**
   - Count and categorize verification steps
   - Statistical analysis of verification methods
   - Summary statistics (mean, median, max, min steps)

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Virtual environment (recommended)

### Installation

1. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the setup script to download required NLTK data:

```bash
python setup.py
```

## Usage

### Running the Analysis

1. Start the application:

```bash
python run.py
```

2. Choose from the following options:
   - Option 1: Load and analyze patient responses from CSV
   - Option 2: Run example analysis
   - Option 3: View debug log
   - Option 4: Exit

### CSV File Format

Your CSV file should contain at least two columns:

- One column for fake news definitions
- One column for verification steps

Example CSV format:

```csv
ID,Definition,VerificationSteps
1,"Fake news is misinformation...","I check multiple sources..."
2,"False information spread...","First I verify the source..."
```

### Analysis Options

#### 1. Loading CSV Data

- Select Option 1 from the main menu
- Choose your CSV file using the file dialog
- Select the appropriate columns for definitions and verification steps

#### 2. Running Example Analysis

- Select Option 2 to run analysis on example data
- Useful for testing and understanding the tool's capabilities

#### 3. Viewing Debug Logs

- Select Option 3 to view detailed debug information
- Helps in troubleshooting and understanding the analysis process

## Coding Scheme

The default coding scheme includes:

```python
coding_scheme = {
    'misinformation': ['false', 'fake', 'lie', 'misinformation'],
    'intentionally deceptive': ['intentionally', 'deliberate', 'purposely'],
    'platform': ['social media', 'internet', 'online'],
    'purpose': ['mislead', 'deceive', 'chaos', 'confusion']
}
```

## Output Format

### Definition Analysis Results

- Coding results (presence/absence of each category)
- Theme frequency analysis
- Common patterns identified

### Verification Steps Analysis

- Number of steps per response
- Statistical summary including:
  - Mean steps
  - Median steps
  - Maximum steps
  - Minimum steps

## Debugging and Logging

### Log Files

- Main log file: `logs/analysis_[timestamp].log`
- Debug log file: `debug_logs/debug_[timestamp].log`

### Log Levels

- DEBUG: Detailed information for debugging
- INFO: General operational information
- WARNING: Warning messages for potential issues
- ERROR: Error messages for actual problems
- CRITICAL: Critical issues that require immediate attention

## Error Handling

The tool includes comprehensive error handling for:

- File loading issues
- Data processing errors
- NLTK-related problems
- Invalid input formats
- Memory issues

## Advanced Features

### Custom Sentence Tokenization

- Primary tokenizer using NLTK's PunktSentenceTokenizer
- Fallback custom tokenizer for handling special cases

### Theme Analysis

- Frequency-based theme identification
- Customizable minimum frequency threshold
- Stop word filtering

## Contributing

To contribute to this tool:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## Troubleshooting

Common issues and solutions:

1. NLTK Data Missing
   - Run `setup.py` again
   - Check internet connection
   - Verify NLTK data directory permissions

2. CSV Loading Issues
   - Verify CSV format
   - Check file encoding (UTF-8 recommended)
   - Ensure column names match expected format

3. Memory Issues
   - Reduce batch size
   - Close other applications
   - Check available system resources

## Support

For issues and questions:

- Check the debug logs
- Review the documentation
- Submit an issue in the repository
