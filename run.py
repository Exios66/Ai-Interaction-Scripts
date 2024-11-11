import sys
import logging
import nltk
from scripts.polyPsych.open_end import OpenEndedAnalysis
from scripts.utils.logging_config import setup_logging

logger = setup_logging()

def initialize_nltk():
    """Initialize NLTK and download required data"""
    try:
        # Download essential NLTK data
        packages = [
            'punkt',
            'stopwords',
            'averaged_perceptron_tagger',
            'wordnet',
            'omw-1.4'
        ]
        
        for package in packages:
            logger.info(f"Downloading NLTK package: {package}")
            try:
                nltk.download(package, quiet=False)  # Set to False to see download progress
            except Exception as e:
                logger.warning(f"Could not download {package}: {str(e)}")
        
        # Verify tokenizer
        from nltk.tokenize import PunktSentenceTokenizer
        tokenizer = PunktSentenceTokenizer()
        test_result = tokenizer.tokenize("Test sentence. Another test.")
        logger.info(f"Tokenizer test successful: {test_result}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize NLTK: {str(e)}")
        return False

def main():
    try:
        # Initialize NLTK
        if not initialize_nltk():
            logger.error("Failed to initialize NLTK. Exiting.")
            sys.exit(1)
        
        # Initialize analyzer
        analyzer = OpenEndedAnalysis()
        
        # Create simple menu
        while True:
            print("\nOpen-Ended Response Analysis")
            print("1. Load and analyze patient responses from CSV")
            print("2. Run example analysis")
            print("3. Exit")
            
            choice = input("Select an option (1-3): ")
            
            if choice == '1':
                if analyzer.load_csv_data():
                    results = analyzer.analyze_patient_responses()
                    if results:
                        print("\nAnalysis Results:")
                        print("\nDefinition Analysis:")
                        print("Coding Results:")
                        print(results['definition_analysis']['coding'])
                        print("\nCommon Themes:")
                        print(results['definition_analysis']['themes'])
                        print("\nVerification Steps Analysis:")
                        print("Steps per Response:")
                        print(results['verification_analysis']['steps'])
                        print("\nSummary Statistics:")
                        for key, value in results['verification_analysis']['summary'].items():
                            print(f"{key}: {value:.2f}")
                    else:
                        print("Analysis failed. Check logs for details.")
                else:
                    print("Failed to load CSV data. Check logs for details.")
                    
            elif choice == '2':
                # Original example analysis code
                responses = [
                    "Fake news is misinformation spread on social media.",
                    "Fake news is news that is intentionally deceptive.",
                    "I think fake news is a lie that people share to create chaos.",
                    "False stories that are made up to mislead the public."
                ]
                
                # Example coding scheme
                coding_scheme = {
                    'misinformation': ['false', 'fake', 'lie', 'misinformation'],
                    'intentionally deceptive': ['intentionally', 'deliberate', 'purposely'],
                    'platform': ['social media', 'internet', 'online'],
                    'purpose': ['mislead', 'deceive', 'chaos', 'confusion']
                }
                
                # Code responses
                coded_df = analyzer.quantify_responses(responses, coding_scheme)
                print("Coded Fake News Responses:")
                print(coded_df)
                
                # Example verification steps
                responses_verification = [
                    "I cross-check the information with reliable news sources.",
                    "First I check the source. Then I verify the date. Finally I look for other coverage.",
                    "Compare with official statements",
                ]
                
                # Analyze verification steps
                steps = analyzer.quantify_verification_steps(responses_verification)
                print("\nVerification Steps per Response:")
                print(steps)
                
                # Analyze themes
                themes = analyzer.analyze_themes(responses)
                print("\nCommon Themes:")
                print(themes)
                
            elif choice == '3':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please select 1-3.")

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 