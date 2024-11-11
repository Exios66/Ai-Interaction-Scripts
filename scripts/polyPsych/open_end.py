import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import logging
import os
import sys
from tkinter import filedialog
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.utils.logging_config import setup_logging
from scripts.utils.debug_utils import (
    function_debug_decorator, 
    DebugContext, 
    log_error_details,
    setup_debug_logging
)

# Initialize logger
logger = setup_logging()

# Add debug logging setup
debug_log_path = setup_debug_logging()

class OpenEndedAnalysis:
    @function_debug_decorator
    def __init__(self):
        """Initialize the OpenEndedAnalysis class with required NLTK downloads"""
        try:
            logger.info("Initializing OpenEndedAnalysis")
            with DebugContext("initialization", logger):
                self._download_nltk_data()
                self.stop_words = set(stopwords.words('english'))
                self._initialize_tokenizer()
                self.data = None
                self.response_columns = None
            logger.info("Initialization complete")
        except Exception as e:
            log_error_details(logger, e, "initialization")
            raise

    @function_debug_decorator
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

    @function_debug_decorator
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

    @function_debug_decorator
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

    @function_debug_decorator
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

    @function_debug_decorator
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

    @function_debug_decorator
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

    @function_debug_decorator
    def load_csv_data(self, file_path=None):
        """
        Load patient response data from CSV
        
        Args:
            file_path (str, optional): Path to CSV file. If None, opens file dialog
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with DebugContext("CSV loading", logger):
                if file_path is None:
                    file_path = filedialog.askopenfilename(
                        title="Select Patient Response CSV File",
                        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                    )
                    
                if not file_path:  # User cancelled
                    return False
                    
                logger.info(f"Loading data from: {file_path}")
                self.data = pd.read_csv(file_path)
                
                # Get column names for responses
                self.select_response_columns()
                
                if self.response_columns is None:
                    logger.error("No response columns selected")
                    return False
                    
                logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
                return True
                
        except Exception as e:
            log_error_details(logger, e, "CSV loading")
            return False

    @function_debug_decorator
    def select_response_columns(self):
        """Open dialog for selecting response columns"""
        if self.data is None:
            return

        root = tk.Tk()
        root.title("Select Response Columns")
        root.geometry("400x300")

        frame = tk.Frame(root)
        frame.pack(padx=10, pady=10)

        tk.Label(frame, text="Select Fake News Definition Column:").pack()
        definition_var = tk.StringVar()
        definition_menu = tk.OptionMenu(frame, definition_var, *self.data.columns)
        definition_menu.pack(pady=5)

        tk.Label(frame, text="Select Verification Steps Column:").pack()
        verification_var = tk.StringVar()
        verification_menu = tk.OptionMenu(frame, verification_var, *self.data.columns)
        verification_menu.pack(pady=5)

        def on_submit():
            self.response_columns = {
                'definition': definition_var.get(),
                'verification': verification_var.get()
            }
            root.destroy()

        tk.Button(frame, text="Submit", command=on_submit).pack(pady=20)
        root.mainloop()

    @function_debug_decorator
    def analyze_patient_responses(self):
        """Analyze loaded patient responses"""
        if self.data is None or self.response_columns is None:
            logger.error("No data loaded or columns selected")
            return None

        try:
            with DebugContext("patient response analysis", logger):
                results = {
                    'definition_analysis': {},
                    'verification_analysis': {}
                }

                # Analyze fake news definitions
                definitions = self.data[self.response_columns['definition']].dropna()
                
                # Code the definitions
                coding_scheme = {
                    'misinformation': ['false', 'fake', 'lie', 'misinformation'],
                    'intentionally deceptive': ['intentionally', 'deliberate', 'purposely'],
                    'platform': ['social media', 'internet', 'online'],
                    'purpose': ['mislead', 'deceive', 'chaos', 'confusion']
                }
                
                results['definition_analysis']['coding'] = self.quantify_responses(
                    definitions.tolist(), 
                    coding_scheme
                )
                
                # Analyze themes in definitions
                results['definition_analysis']['themes'] = self.analyze_themes(
                    definitions.tolist()
                )

                # Analyze verification steps
                verifications = self.data[self.response_columns['verification']].dropna()
                results['verification_analysis']['steps'] = self.quantify_verification_steps(
                    verifications.tolist()
                )
                
                # Calculate summary statistics
                results['verification_analysis']['summary'] = {
                    'mean_steps': np.mean(results['verification_analysis']['steps']),
                    'median_steps': np.median(results['verification_analysis']['steps']),
                    'max_steps': max(results['verification_analysis']['steps']),
                    'min_steps': min(results['verification_analysis']['steps'])
                }

                logger.info("Patient response analysis completed successfully")
                return results

        except Exception as e:
            log_error_details(logger, e, "patient response analysis")
            return None

    @function_debug_decorator
    def export_results(self, results, export_dir='exports'):
        """Export analysis results to various formats in a timestamped subfolder"""
        try:
            # Create main export directory if it doesn't exist
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            # Create timestamped subfolder
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_subfolder = os.path.join(export_dir, timestamp)
            os.makedirs(export_subfolder)
            
            logger.info(f"Created export subfolder: {export_subfolder}")
            
            # Export metadata
            metadata = {
                'export_timestamp': timestamp,
                'analysis_type': 'Open Ended Response Analysis',
                'num_responses': len(self.data) if self.data is not None else 0,
                'export_contents': []
            }
            
            # Export coded responses to CSV
            if 'definition_analysis' in results:
                coded_df = results['definition_analysis']['coding']
                csv_path = os.path.join(export_subfolder, 'coded_responses.csv')
                coded_df.to_csv(csv_path, index=False)
                metadata['export_contents'].append('coded_responses.csv')
                logger.info(f"Exported coded responses to: {csv_path}")

            # Export themes to JSON
            if 'definition_analysis' in results and 'themes' in results['definition_analysis']:
                themes_path = os.path.join(export_subfolder, 'themes.json')
                with open(themes_path, 'w') as f:
                    json.dump(results['definition_analysis']['themes'], f, indent=4)
                metadata['export_contents'].append('themes.json')
                logger.info(f"Exported themes to: {themes_path}")

            # Export verification analysis to CSV
            if 'verification_analysis' in results:
                verif_df = pd.DataFrame({
                    'steps': results['verification_analysis']['steps']
                })
                verif_path = os.path.join(export_subfolder, 'verification_analysis.csv')
                verif_df.to_csv(verif_path, index=False)
                metadata['export_contents'].append('verification_analysis.csv')
                logger.info(f"Exported verification analysis to: {verif_path}")

            # Export summary statistics
            if 'verification_analysis' in results and 'summary' in results['verification_analysis']:
                stats_path = os.path.join(export_subfolder, 'summary_stats.json')
                with open(stats_path, 'w') as f:
                    json.dump(results['verification_analysis']['summary'], f, indent=4)
                metadata['export_contents'].append('summary_stats.json')
                logger.info(f"Exported summary statistics to: {stats_path}")

            # Export metadata
            metadata_path = os.path.join(export_subfolder, 'export_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Exported metadata to: {metadata_path}")

            # Create README file
            readme_path = os.path.join(export_subfolder, 'README.txt')
            with open(readme_path, 'w') as f:
                f.write(f"Open Ended Response Analysis Export\n")
                f.write(f"Generated: {timestamp}\n\n")
                f.write("Contents:\n")
                for file in metadata['export_contents']:
                    f.write(f"- {file}\n")
                f.write("\nFor questions or support, please contact the development team.")
            
            logger.info(f"Export completed successfully to: {export_subfolder}")
            return True, export_subfolder
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}", exc_info=True)
            return False, None

    @function_debug_decorator
    def create_visualizations(self, results, export_dir='exports'):
        """Create and export data visualizations to the same timestamped subfolder"""
        try:
            # Use the most recent export subfolder if it exists
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
                
            # Find most recent subfolder
            subfolders = [f for f in os.listdir(export_dir) 
                         if os.path.isdir(os.path.join(export_dir, f))]
            if not subfolders:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                viz_subfolder = os.path.join(export_dir, timestamp)
                os.makedirs(viz_subfolder)
            else:
                latest_subfolder = max(subfolders)
                viz_subfolder = os.path.join(export_dir, latest_subfolder)
            
            # Create visualizations subfolder
            viz_folder = os.path.join(viz_subfolder, 'visualizations')
            os.makedirs(viz_folder, exist_ok=True)
            
            # Theme frequency visualization
            if 'definition_analysis' in results and 'themes' in results['definition_analysis']:
                plt.figure(figsize=(12, 6))
                themes = results['definition_analysis']['themes']
                plt.bar(themes.keys(), themes.values())
                plt.title('Theme Frequencies in Responses')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                theme_plot_path = os.path.join(viz_folder, 'theme_frequencies.png')
                plt.savefig(theme_plot_path)
                plt.close()
                logger.info(f"Theme frequency plot saved to: {theme_plot_path}")

            # Verification steps distribution
            if 'verification_analysis' in results and 'steps' in results['verification_analysis']:
                plt.figure(figsize=(10, 6))
                steps = results['verification_analysis']['steps']
                sns.histplot(steps, bins=max(steps)-min(steps)+1)
                plt.title('Distribution of Verification Steps')
                plt.xlabel('Number of Steps')
                plt.ylabel('Frequency')
                steps_plot_path = os.path.join(viz_folder, 'verification_steps.png')
                plt.savefig(steps_plot_path)
                plt.close()
                logger.info(f"Verification steps plot saved to: {steps_plot_path}")

            # Coding categories visualization
            if 'definition_analysis' in results and 'coding' in results['definition_analysis']:
                plt.figure(figsize=(10, 6))
                coding_sums = results['definition_analysis']['coding'].sum()
                plt.bar(coding_sums.index, coding_sums.values)
                plt.title('Frequency of Coding Categories')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                coding_plot_path = os.path.join(viz_folder, 'coding_frequencies.png')
                plt.savefig(coding_plot_path)
                plt.close()
                logger.info(f"Coding frequencies plot saved to: {coding_plot_path}")

            # Update metadata to include visualizations
            metadata_path = os.path.join(viz_subfolder, 'export_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata['visualizations'] = [
                    'visualizations/theme_frequencies.png',
                    'visualizations/verification_steps.png',
                    'visualizations/coding_frequencies.png'
                ]
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)

            logger.info(f"Visualizations created successfully in: {viz_folder}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}", exc_info=True)
            return False

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
