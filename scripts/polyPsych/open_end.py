import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import logging
import os
import sys

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.utils.logging_config import setup_logging

# Initialize logger
logger = setup_logging()

class OpenEndedAnalysis:
    def __init__(self):
        """Initialize the OpenEndedAnalysis class with required NLTK downloads"""
        try:
            logger.info("Initializing OpenEndedAnalysis")
            self._download_nltk_data()
            self.stop_words = set(stopwords.words('english'))
            self._initialize_tokenizer()
            logger.info("Initialization complete")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            raise

    def _download_nltk_data(self):
        """Download required NLTK data packages"""
        try:
            # Define all required NLTK packages
            required_packages = [
                'punkt',
                'stopwords',
                'averaged_perceptron_tagger',
                'wordnet',
                'omw-1.4'
            ]
            
            # Download each package
            for package in required_packages:
                try:
                    logger.debug(f"Checking/Downloading NLTK package: {package}")
                    nltk.download(package, quiet=False)  # Set to False to see download progress
                except Exception as e:
                    logger.error(f"Failed to download {package}: {str(e)}")
                    raise
            
            # Verify punkt tokenizer is available
            try:
                test_result = sent_tokenize("Test sentence. Another test.")
                logger.info(f"Tokenizer test successful: {test_result}")
            except LookupError:
                logger.warning("Punkt tokenizer not found, downloading additional resources...")
                nltk.download('punkt', quiet=False)
            
            logger.info("All required NLTK packages downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download NLTK packages: {str(e)}", exc_info=True)
            raise

    def _initialize_tokenizer(self):
        """Initialize the sentence tokenizer"""
        try:
            from nltk.tokenize import PunktSentenceTokenizer
            self.sentence_tokenizer = PunktSentenceTokenizer()
            # Test the tokenizer
            self.sentence_tokenizer.tokenize("Test sentence. Another test.")
            logger.info("Sentence tokenizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize sentence tokenizer: {str(e)}")
            raise

    def _custom_sentence_split(self, text):
        """Fallback method for sentence splitting"""
        # Simple sentence splitting based on common punctuation
        text = str(text)
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?' and len(current.strip()) > 0:
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
            
        return sentences

    def quantify_responses(self, responses, coding_scheme):
        """
        Quantify open-ended responses using a predefined coding scheme
        
        Args:
            responses (list): List of text responses
            coding_scheme (dict): Dictionary of categories and their keywords
        
        Returns:
            pd.DataFrame: Coded responses
        """
        try:
            logger.info(f"Processing {len(responses)} responses")
            results = []
            
            for response in responses:
                logger.debug(f"Processing response: '{response}'")
                coded_response = {}
                
                # Convert response to lowercase for matching
                response_lower = str(response).lower()
                
                # Code each category
                for category, keywords in coding_scheme.items():
                    # Check if any keyword is present in the response
                    coded_response[category] = int(
                        any(keyword in response_lower for keyword in keywords)
                    )
                
                results.append(coded_response)
                logger.debug(f"Coding result for response: {coded_response}")
            
            # Convert results to DataFrame
            coded_df = pd.DataFrame(results)
            logger.info("Response coding completed successfully")
            return coded_df
            
        except Exception as e:
            logger.error(f"Error in quantifying responses: {str(e)}", exc_info=True)
            raise

    def quantify_verification_steps(self, responses):
        """
        Quantify verification steps in responses by counting sentences
        
        Args:
            responses (list): List of verification step descriptions
            
        Returns:
            list: Number of verification steps per response
        """
        try:
            logger.info("Quantifying verification steps")
            verification_steps = []
            
            for response in responses:
                logger.debug(f"Processing response for verification steps: '{response}'")
                if pd.isna(response) or not response:
                    steps = 0
                else:
                    try:
                        # Try using the initialized tokenizer
                        sentences = self.sentence_tokenizer.tokenize(str(response))
                    except Exception as e:
                        logger.warning(f"Primary tokenizer failed, using fallback: {str(e)}")
                        # Fallback to custom sentence splitting
                        sentences = self._custom_sentence_split(str(response))
                    
                    steps = len(sentences)
                
                verification_steps.append(steps)
                logger.debug(f"Identified {steps} steps in response")
            
            logger.info("Verification step quantification completed")
            return verification_steps
            
        except Exception as e:
            logger.error(f"Error in quantifying verification steps: {str(e)}", exc_info=True)
            raise

    def analyze_themes(self, responses, min_freq=2):
        """
        Analyze common themes in responses using word frequency
        
        Args:
            responses (list): List of text responses
            min_freq (int): Minimum frequency for word inclusion
            
        Returns:
            dict: Word frequencies above minimum threshold
        """
        try:
            logger.info("Analyzing response themes")
            word_freq = {}
            
            for response in responses:
                if pd.isna(response) or not response:
                    continue
                    
                # Tokenize and process words
                words = word_tokenize(str(response).lower())
                words = [word for word in words if word.isalnum() 
                        and word not in self.stop_words]
                
                # Count word frequencies
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Filter by minimum frequency
            themes = {word: freq for word, freq in word_freq.items() 
                     if freq >= min_freq}
            
            logger.info(f"Theme analysis complete. Found {len(themes)} themes")
            return themes
            
        except Exception as e:
            logger.error(f"Error in theme analysis: {str(e)}", exc_info=True)
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Initialize analyzer
        analyzer = OpenEndedAnalysis()
        
        # Example responses
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
        
    except Exception as e:
        logger.critical(f"Application failed: {str(e)}", exc_info=True)
        raise
