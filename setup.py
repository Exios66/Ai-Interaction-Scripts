import nltk
import sys
import os

def setup_nltk():
    """Setup NLTK data in the correct location"""
    try:
        # Create NLTK data directory if it doesn't exist
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)

        # Download all required NLTK data with correct package names
        packages = [
            'punkt',
            'stopwords',
            'averaged_perceptron_tagger',
            'wordnet',
            'omw-1.4'  # Required for newer versions of NLTK
        ]
        
        for package in packages:
            print(f"Downloading {package}...")
            try:
                nltk.download(package, quiet=False)  # Set to False to see download progress
            except Exception as e:
                print(f"Warning: Could not download {package}: {str(e)}")
                continue
        
        # Verify punkt tokenizer
        try:
            from nltk.tokenize import PunktSentenceTokenizer
            tokenizer = PunktSentenceTokenizer()
            test_result = tokenizer.tokenize("Test sentence. Another test.")
            print(f"Punkt tokenizer verified successfully: {test_result}")
        except Exception as e:
            print(f"Warning: Punkt tokenizer verification failed: {str(e)}")
            # Try alternative download method
            try:
                nltk.download('punkt', download_dir=nltk_data_dir)
                print("Punkt downloaded using alternative method")
            except Exception as sub_e:
                print(f"Alternative download also failed: {str(sub_e)}")
        
        print("NLTK setup completed")
        return True
    except Exception as e:
        print(f"Error during NLTK setup: {str(e)}")
        return False

if __name__ == "__main__":
    if not setup_nltk():
        sys.exit(1)