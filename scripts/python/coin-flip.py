import random
import re
import os
import csv
from collections import Counter
import matplotlib.pyplot as plt

# Add this constant near the top of the file, after imports
DEFAULT_FACTS = [
    "The Great Wall of China is over 13,000 miles long",
    "The human body has 206 bones",
    "Mount Everest grows about 4 millimeters per year",
    "A honeybee visits around 5,000 flowers in a single day",
    "The speed of light is 299,792 kilometers per second",
    "The Earth's core temperature is about 6,000 degrees Celsius",
    "The average adult human brain weighs 3 pounds",
    "Dolphins sleep with one half of their brain at a time",
    "A day on Venus is longer than its year",
    "The first computer programmer was a woman named Ada Lovelace",
    "The human eye can distinguish about 10 million different colors",
    "Honey never spoils if stored properly",
    "Octopuses have three hearts",
    "The longest recorded flight of a chicken is 13 seconds",
    "The shortest war in history lasted 38 minutes"
]

def flip_coin():
    """
    Returns True or False based on a 50/50 coin flip.
    """
    # Randomly select between True and False to simulate a fair coin flip
    return random.choice([True, False])

def alter_fact(fact):
    """
    Alters the provided fact to create a false version of it.
    The alteration is more sophisticated and works by:
    1. Switching key words/values
    2. Adding qualifiers
    3. Changing numbers if present
    """
    if not isinstance(fact, str):
        return "Invalid fact format"
        
    # List of word substitutions to create uncertainty
    uncertainty_words = {
        r'\b(is|are)\b': 'might be',
        r'\b(was|were)\b': 'might have been', 
        r'\b(has|have|had)\b': 'may have',
        r'\b(can|could)\b': 'possibly can',
        r'\b(will|would)\b': 'might'
    }
    
    altered_fact = fact
    substitution_made = False
    
    # Apply word substitutions
    for pattern, replacement in uncertainty_words.items():
        if re.search(pattern, altered_fact):
            altered_fact = re.sub(pattern, replacement, altered_fact, count=1)
            substitution_made = True
            break
            
    # Change numbers if present
    number_pattern = r'\b\d+\b'
    if re.search(number_pattern, altered_fact):
        def modify_number(match):
            num = int(match.group(0))
            # Modify number by ±10-50%
            modification = random.uniform(0.5, 1.5)
            return str(round(num * modification))
            
        altered_fact = re.sub(number_pattern, modify_number, altered_fact)
        substitution_made = True
        
    # If no substitutions were made, add a qualifier
    if not substitution_made:
        qualifiers = [
            "Some experts dispute that",
            "It's debatable whether",
            "There's uncertainty about whether",
            "Recent studies question if"
        ]
        altered_fact = f"{random.choice(qualifiers)} {altered_fact.lower()}"
        
    return altered_fact

def present_facts(facts, num_facts):
    """
    Presents a specified number of facts from the list with improved formatting.
    
    Args:
    - facts (list): The list containing the facts
    - num_facts (int): The number of facts to present
    """
    if not facts:
        print("Error: No facts available")
        return
        
    if num_facts > len(facts):
        print(f"Warning: Requested {num_facts} facts but only {len(facts)} are available")
        num_facts = len(facts)
        
    # Randomly select the specified number of facts
    selected_facts = random.sample(facts, num_facts)
    
    print("\n=== FACT CHECKING GAME ===\n")
    correct_guesses = 0
    total_facts = 0
    
    for fact in selected_facts:
        total_facts += 1
        fact = str(fact).strip()
        if not fact:  # Skip empty facts
            continue
            
        # Flip a coin to decide whether to present the fact truthfully or falsely
        is_true = flip_coin()
        presented_fact = fact if is_true else alter_fact(fact)
        
        # Present the fact and get user's guess
        print(f"\nFact #{total_facts}:")
        print(f"{presented_fact}")
        
        while True:
            guess = input("\nIs this fact True or False? (T/F): ").strip().upper()
            if guess in ['T', 'F']:
                break
            print("Please enter 'T' for True or 'F' for False")
            
        # Check if the guess was correct
        is_correct = (guess == 'T' and is_true) or (guess == 'F' and not is_true)
        if is_correct:
            correct_guesses += 1
            print("Correct! ✓")
        else:
            print("Incorrect! ✗")
            print(f"The original fact was: {fact}")
            
    # Display final score
    print(f"\nFinal Score: {correct_guesses}/{total_facts} correct")
    percentage = (correct_guesses / total_facts) * 100
    print(f"Accuracy: {percentage:.1f}%")

def load_questions_file():
    """
    Loads the appropriate questions CSV file based on user input.
    Now includes an option for using default embedded facts.
    
    Returns:
    - tuple: (filename or None, use_default_facts boolean)
    """
    print("\nHow would you like to play?")
    print("1. Use built-in facts")
    print("2. Load facts from a CSV file")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        return None, True
        
    if choice == "2":
        # Predefined list of available files
        available_files = [
            'psychology_questions.csv',
            'literature_questions.csv',
            'politics_questions.csv',
            'american_history_questions.csv',
            'chemistry_questions.csv'
        ]
        
        # Add any other CSV files in the current directory to the available files list
        try:
            for file in os.listdir():
                if file.endswith('.csv') and file not in available_files:
                    available_files.append(file)
        except OSError as e:
            print(f"Error accessing directory: {e}")
            return None, True
            
        if not available_files:
            print("Error: No question files found")
            return None, True
            
        # Display available files to the user
        print("\nAvailable question files:")
        for i, file in enumerate(available_files, start=1):
            print(f"{i}. {file}")
        
        max_attempts = 3
        attempts = 0
        
        while attempts < max_attempts:
            try:
                file_choice = int(input("\nEnter the number corresponding to the file you want to use: "))
                if 1 <= file_choice <= len(available_files):
                    return available_files[file_choice - 1], False
                else:
                    attempts += 1
                    remaining = max_attempts - attempts
                    print(f"Invalid choice. Please enter a number between 1 and {len(available_files)}.")
                    if remaining > 0:
                        print(f"Attempts remaining: {remaining}")
            except ValueError:
                attempts += 1
                remaining = max_attempts - attempts
                print("Invalid input. Please enter a number.")
                if remaining > 0:
                    print(f"Attempts remaining: {remaining}")
                    
        print("Maximum attempts reached. Exiting program.")
        return None, True
    
    print("Invalid choice or maximum attempts reached. Using built-in facts.")
    return None, True

def run_coin_flip_analysis(num_flips):
    """
    Runs multiple coin flips and provides statistical analysis.
    
    Args:
    - num_flips (int): Number of coin flips to simulate
    
    Returns:
    - tuple: (results list, statistics dict)
    """
    results = []
    for _ in range(num_flips):
        flip_result = flip_coin()
        results.append("TRUE" if flip_result else "MISLEAD")
    
    # Calculate statistics
    counts = Counter(results)
    stats = {
        'total_flips': num_flips,
        'true_count': counts["TRUE"],
        'mislead_count': counts["MISLEAD"],
        'true_percentage': (counts["TRUE"] / num_flips) * 100,
        'mislead_percentage': (counts["MISLEAD"] / num_flips) * 100
    }
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.bar(['TRUE', 'MISLEAD'], [counts["TRUE"], counts["MISLEAD"]], 
            color=['green', 'red'])
    plt.title(f'Coin Flip Results ({num_flips} flips)')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels on top of bars
    for i, count in enumerate([counts["TRUE"], counts["MISLEAD"]]):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.show()
    
    return results, stats

def main():
    print("\nWelcome to the Fact Checking Game!")
    print("Choose your mode:")
    print("1. Play Fact Checking Game")
    print("2. Run Coin Flip Analysis")
    
    mode = input("\nEnter your choice (1 or 2): ").strip()
    
    if mode == "2":
        while True:
            print("\nHow many coin flips would you like to simulate?")
            print("1. 10 flips")
            print("2. 100 flips")
            print("3. 1000 flips")
            print("4. Custom number")
            print("5. Return to main menu")
            
            flip_choice = input("\nEnter your choice (1-5): ").strip()
            
            flip_counts = {
                "1": 10,
                "2": 100,
                "3": 1000
            }
            
            if flip_choice in flip_counts:
                results, stats = run_coin_flip_analysis(flip_counts[flip_choice])
                print("\nAnalysis Results:")
                print(f"Total Flips: {stats['total_flips']}")
                print(f"TRUE Results: {stats['true_count']} ({stats['true_percentage']:.1f}%)")
                print(f"MISLEAD Results: {stats['mislead_count']} ({stats['mislead_percentage']:.1f}%)")
                input("\nPress Enter to continue...")
                
            elif flip_choice == "4":
                try:
                    custom_count = int(input("Enter number of flips (1-10000): "))
                    if 1 <= custom_count <= 10000:
                        results, stats = run_coin_flip_analysis(custom_count)
                        print("\nAnalysis Results:")
                        print(f"Total Flips: {stats['total_flips']}")
                        print(f"TRUE Results: {stats['true_count']} ({stats['true_percentage']:.1f}%)")
                        print(f"MISLEAD Results: {stats['mislead_count']} ({stats['mislead_percentage']:.1f}%)")
                        input("\nPress Enter to continue...")
                    else:
                        print("Please enter a number between 1 and 10000")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    
            elif flip_choice == "5":
                main()
                return
            else:
                print("Invalid choice. Please select 1-5.")
    
    elif mode == "1":
        # Original game code
        questions_file, use_defaults = load_questions_file()
        # ... (rest of the original game code)
    
    else:
        print("Invalid choice. Please select 1 or 2.")
        main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
