# AI Interaction Scripts Repository

## 📋 Overview

A comprehensive collection of AI interaction and analysis tools, focusing on psychological clustering and open-ended response analysis. This repository provides tools for analyzing survey responses, performing psychological clustering, and processing qualitative data.

## 🚀 Key Features

### Open-Ended Response Analysis

- Process and analyze survey responses
- Automated coding of qualitative data
- Theme identification and analysis
- Verification step quantification
- Statistical summaries and reporting

### Psychological Clustering

- Advanced clustering algorithms
- Pattern recognition
- Data visualization
- Statistical analysis

### Core Capabilities

- CSV data import/export
- Natural Language Processing
- Automated theme detection
- Statistical analysis
- Comprehensive logging and debugging
- GUI interfaces for data selection

## 📁 Repository Structure

```bash
AI-Interaction-Scripts/
├── scripts/
│   ├── README.md                 # Detailed script documentation
│   ├── __init__.py              # Package initialization
│   ├── polyPsych/               # Psychological analysis modules
│   │   ├── clustering.py        # Clustering algorithms
│   │   └── open_end.py         # Open-ended response analysis
│   └── utils/                   # Utility modules
│       ├── __init__.py         # Utils initialization
│       ├── debug_utils.py      # Debugging utilities
│       └── logging_config.py   # Logging configuration
├── logs/                        # Log file directory
│   ├── analysis_*.log          # Analysis logs
│   └── debug_*.log            # Debug logs
├── debug_logs/                  # Detailed debug information
├── requirements.txt            # Project dependencies
├── setup.py                    # NLTK setup script
├── run.py                      # Main execution script
└── README.md                   # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Steps

1. Clone the repository:

```bash
git clone https://github.com/yourusername/AI-Interaction-Scripts.git
cd AI-Interaction-Scripts
```

2. Create and activate virtual environment:

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run NLTK setup:

```bash
python setup.py
```

## 💻 Usage

### Running the Analysis Tool

1. Start the analysis tool:

```bash
python run.py
```

2. Choose from available options:
   - Load and analyze CSV responses
   - Run example analysis
   - View debug logs
   - Exit

### Data Requirements

#### CSV Format for Response Analysis

```csv
ID,Definition,VerificationSteps
1,"Response text...","Verification steps..."
```

### Analysis Features

1. **Response Coding**
   - Automated categorization
   - Theme identification
   - Frequency analysis

2. **Verification Analysis**
   - Step counting
   - Statistical summaries
   - Pattern identification

3. **Theme Analysis**
   - Keyword extraction
   - Frequency analysis
   - Pattern recognition

## 🔍 Debugging and Logging

### Log Files

- Analysis logs: `logs/analysis_[timestamp].log`
- Debug logs: `debug_logs/debug_[timestamp].log`
- Error tracking: Comprehensive stack traces

### Debug Levels

- DEBUG: Detailed execution information
- INFO: General operational messages
- WARNING: Potential issues
- ERROR: Operation failures
- CRITICAL: System-critical issues

## 🛠 Advanced Features

### Custom Tokenization

- NLTK-based processing
- Fallback mechanisms
- Custom sentence splitting

### Statistical Analysis

- Descriptive statistics
- Frequency analysis
- Pattern recognition
- Clustering analysis

## 📊 Output Formats

### Analysis Results

- Coded responses
- Theme frequencies
- Statistical summaries
- Verification patterns

### Export Options

- CSV format
- JSON data
- Statistical reports
- Debug logs

## 🐛 Troubleshooting

### Common Issues

1. NLTK Data
   - Run `setup.py`
   - Check internet connection
   - Verify data directory

2. Data Loading
   - Check CSV format
   - Verify encoding (UTF-8)
   - Column name matching

3. System Resources
   - Memory management
   - Process optimization
   - Resource allocation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes
4. Submit pull request

### Development Guidelines

- Follow PEP 8 style guide
- Add comprehensive documentation
- Include unit tests
- Update README as needed

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📫 Support

- Review documentation
- Check debug logs
- Submit issues
- Contact maintainers

## 🙏 Acknowledgments

- NLTK Project
- Python Data Science Community
- Open Source Contributors

---

Made with ❤️ by [Your Name]

For detailed script-specific documentation, see [scripts/README.md](scripts/README.md)
