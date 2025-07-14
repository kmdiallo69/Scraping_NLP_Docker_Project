#!/usr/bin/env python3
"""
ToxiGuard French Tweets - Helper Functions Module

This module contains utility functions for text processing, data cleaning,
file management, and dataset preparation for the toxicity detection system.

Author: ToxiGuard Team
License: MIT
"""

# Standard library imports
import os
import re
import glob
from typing import Dict, List, Optional

# Third-party imports
import pandas as pd
from detoxify import Detoxify


# =============================================================================
# TEXT CLEANING CONFIGURATION
# =============================================================================

# Unicode character mappings for French text normalization
# This dictionary maps common Unicode characters to their ASCII equivalents
UNICODE_CHAR_MAP: Dict[str, str] = {
    # Lowercase accented vowels
    '\\u00e0': 'a',  # à
    '\\u00e2': 'a',  # â
    '\\u00e4': 'a',  # ä
    '\\u00e8': 'e',  # è
    '\\u00e9': 'e',  # é
    '\\u00ea': 'e',  # ê
    '\\u00eb': 'e',  # ë
    '\\u00ee': 'i',  # î
    '\\u00ef': 'i',  # ï
    '\\u00f4': 'o',  # ô
    '\\u00f6': 'o',  # ö
    '\\u00f9': 'u',  # ù
    '\\u00fb': 'u',  # û
    '\\u00fc': 'u',  # ü
    '\\u00e7': 'c',  # ç
    
    # Uppercase accented vowels
    '\\u00C0': 'A',  # À
    '\\u00C1': 'A',  # Á
    '\\u00C2': 'A',  # Â
    '\\u00C3': 'A',  # Ã
    '\\u00c0': 'A',  # À (duplicate)
    '\\u00c8': 'E',  # È
    '\\u00C9': 'E',  # É
    '\\u00c9': 'E',  # É (duplicate)
    '\\u00CA': 'E',  # Ê
    '\\u00CB': 'E',  # Ë
    '\\u00CC': 'I',  # Ì
    '\\u00CD': 'I',  # Í
    '\\u00CE': 'I',  # Î
    '\\u00CF': 'I',  # Ï
    '\\u00D2': 'O',  # Ò
    '\\u00D3': 'O',  # Ó
    '\\u00D4': 'O',  # Ô
    '\\u00D5': 'O',  # Õ
    '\\u00D6': 'O',  # Ö
    '\\u00D9': 'U',  # Ù
    '\\u00DA': 'U',  # Ú
    '\\u00DB': 'U',  # Û
    '\\u00DC': 'U',  # Ü
    '\\u00c7': 'C',  # Ç
    
    # Special characters and symbols
    '\\u2019': "'",  # Right single quotation mark
    '\\u2018': "'",  # Left single quotation mark
    '\\u20ac': "euros",  # Euro symbol
    '\\u00a0': '',   # Non-breaking space
    '\\u00ab': "'",  # Left-pointing double angle quotation mark
    '\\u00bb': "'",  # Right-pointing double angle quotation mark
    
    # HTML entities
    '&lt;': 'inferieur',       # Less than
    '&le;': 'inferieur egale', # Less than or equal
    '&gt;': 'superieur',       # Greater than
    '&ge;': 'superieur egale', # Greater than or equal
    
    # Emoji patterns (simplified removal)
    '\\ud83d': '',  # Emoji prefix
    '\\ude09': '',  # Specific emoji codes
    '\\ude18': '',
    '\\ude08': '',
    
    # Punctuation and formatting characters
    ',': ' ',   # Replace comma with space
    '\n': ' ',  # Replace newline with space
    '\r': '',   # Remove carriage return
    '//': '',   # Remove double slash
    '"': '',    # Remove quotes
    '*': '',    # Remove asterisk
    '(': '',    # Remove parentheses
    ')': '',
    '?': '',    # Remove question mark
    '.': '',    # Remove period
    '!': '',    # Remove exclamation mark
    '+': '',    # Remove plus sign
    '-': '',    # Remove hyphen
    '[': '',    # Remove square brackets
    ']': '',
    '{': '',    # Remove curly braces
    '}': ''
}


# =============================================================================
# TOXICITY DETECTION MODEL INITIALIZATION
# =============================================================================

# Initialize the Detoxify model for toxicity detection
# Using the multilingual model for better French language support
print("🤖 Loading Detoxify multilingual model...")
try:
    TOXICITY_MODEL = Detoxify('multilingual')
    print("✅ Detoxify model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Detoxify model: {e}")
    TOXICITY_MODEL = None


# =============================================================================
# DIRECTORY MANAGEMENT FUNCTIONS
# =============================================================================

def ensure_directory_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path (str): Path to the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"📁 Created directory: {directory_path}")
    else:
        print(f"✅ Directory already exists: {directory_path}")


def setup_project_directories() -> None:
    """
    Create all necessary project directories.
    """
    directories = ['./data', './fastapi/model', './logs']
    
    print("📁 Setting up project directories...")
    for directory in directories:
        ensure_directory_exists(directory)
    print("✅ All directories are ready")


# =============================================================================
# TEXT CLEANING FUNCTIONS
# =============================================================================

def clean_text(text: str) -> str:
    """
    Comprehensive text cleaning function for French tweets.
    
    This function performs multiple cleaning operations:
    1. Unicode character normalization
    2. URL and mention removal
    3. Hashtag processing
    4. Number removal
    5. Case normalization
    6. Whitespace cleanup
    
    Args:
        text (str): Raw tweet text to clean
        
    Returns:
        str: Cleaned and normalized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Replace Unicode characters with ASCII equivalents
    for unicode_char, replacement in UNICODE_CHAR_MAP.items():
        text = text.replace(unicode_char, replacement)
    
    # Step 2: Remove hashtag symbols (but keep the text)
    text = re.sub('#', '', text)
    
    # Step 3: Remove Twitter user mentions (@username)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    
    # Step 4: Remove URLs (http/https and www links)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    
    # Step 5: Remove numbers and number-containing words
    text = re.sub(r"\d\S*", '', text)
    
    # Step 6: Convert to lowercase for consistency
    text = text.lower()
    
    # Step 7: Remove extra whitespace and normalize spaces
    text = text.strip()  # Remove leading/trailing whitespace
    text = re.sub(r' +', ' ', text)  # Replace multiple spaces with single space
    
    return text


def batch_clean_texts(texts: List[str]) -> List[str]:
    """
    Clean a list of texts efficiently.
    
    Args:
        texts (List[str]): List of raw texts to clean
        
    Returns:
        List[str]: List of cleaned texts
    """
    print(f"🧹 Cleaning {len(texts)} texts...")
    
    cleaned_texts = []
    for i, text in enumerate(texts):
        cleaned = clean_text(text)
        if cleaned:  # Only add non-empty cleaned texts
            cleaned_texts.append(cleaned)
        
        # Progress indicator for large batches
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1}/{len(texts)} texts...")
    
    print(f"✅ Cleaning complete: {len(cleaned_texts)} valid texts from {len(texts)} original")
    return cleaned_texts


# =============================================================================
# FILE MANAGEMENT FUNCTIONS
# =============================================================================

def merge_csv_files() -> None:
    """
    Merge all CSV files in the data directory into a single file.
    
    This function:
    1. Finds all CSV files in the data directory
    2. Combines them into a single DataFrame
    3. Removes the original individual files
    4. Saves the merged data as 'merge_csv.csv'
    """
    print("🔄 Starting CSV file merge process...")
    
    # Find all CSV files in the data directory
    csv_files = glob.glob('./data/*.csv')
    
    if not csv_files:
        print("⚠️ No CSV files found in data directory")
        return
    
    print(f"📋 Found {len(csv_files)} CSV files to merge:")
    for csv_file in csv_files:
        print(f"   📄 {os.path.basename(csv_file)}")
    
    # Initialize empty DataFrame with consistent structure
    merged_df = pd.DataFrame({'text': []})
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Read the CSV file
            temp_df = pd.read_csv(csv_file)
            
            # Ensure it has the expected 'text' column
            if 'text' not in temp_df.columns:
                print(f"⚠️ Skipping {csv_file}: missing 'text' column")
                continue
            
            # Add to merged DataFrame
            merged_df = pd.concat([merged_df, temp_df[['text']]], ignore_index=True)
            
            # Remove the processed file
            os.remove(csv_file)
            print(f"✅ Processed and removed: {os.path.basename(csv_file)}")
            
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")
            continue
    
    # Remove duplicate entries
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset='text')
    final_count = len(merged_df)
    
    print(f"🔍 Removed {initial_count - final_count} duplicate entries")
    
    # Save the merged file
    output_path = './data/merge_csv.csv'
    merged_df.to_csv(output_path, index=False)
    
    print(f"💾 Merged dataset saved: {output_path}")
    print(f"📊 Total unique tweets: {final_count}")


def clean_merged_file() -> None:
    """
    Clean the merged CSV file using the text cleaning function.
    
    This function:
    1. Loads the merged CSV file
    2. Applies text cleaning to all entries
    3. Removes empty/invalid entries
    4. Saves the cleaned data back to the file
    """
    print("🧹 Starting merged file cleaning process...")
    
    input_file = './data/merge_csv.csv'
    
    # Check if the merged file exists
    if not os.path.exists(input_file):
        print(f"❌ Merged file not found: {input_file}")
        print("💡 Run merge_csv_files() first")
        return
    
    try:
        # Load the merged dataset
        print(f"📂 Loading data from: {input_file}")
        df = pd.read_csv(input_file)
        
        print(f"📊 Original dataset: {len(df)} entries")
        
        # Apply text cleaning to all entries
        print("🧹 Applying text cleaning...")
        df['text'] = df['text'].apply(clean_text)
        
        # Remove empty or null entries after cleaning
        initial_count = len(df)
        df = df[df['text'].notna() & (df['text'] != '')]
        final_count = len(df)
        
        print(f"🗑️ Removed {initial_count - final_count} empty entries after cleaning")
        
        # Remove the original file and save the cleaned version
        os.remove(input_file)
        df.to_csv(input_file, index=False)
        
        print(f"✅ Cleaned dataset saved: {input_file}")
        print(f"📊 Final dataset: {final_count} valid entries")
        
    except Exception as e:
        print(f"❌ Error during file cleaning: {e}")


# =============================================================================
# TOXICITY LABELING FUNCTIONS
# =============================================================================

def predict_toxicity(text: str) -> str:
    """
    Predict toxicity label for a given text using Detoxify model.
    
    Args:
        text (str): Text to analyze for toxicity
        
    Returns:
        str: 'toxic' if toxic content detected, 'no_toxic' otherwise
    """
    if not TOXICITY_MODEL:
        print("❌ Toxicity model not available")
        return 'no_toxic'
    
    if not text or not isinstance(text, str):
        return 'no_toxic'
    
    try:
        # Get toxicity predictions from Detoxify
        predictions = TOXICITY_MODEL.predict(text)
        
        # Find the highest confidence prediction
        max_category = max(predictions, key=predictions.get)
        max_confidence = predictions[max_category]
        
        # Use threshold of 0.3 for toxicity classification
        # This threshold can be adjusted based on validation results
        toxicity_threshold = 0.3
        
        if max_confidence > toxicity_threshold:
            return 'toxic'
        else:
            return 'no_toxic'
            
    except Exception as e:
        print(f"⚠️ Error in toxicity prediction: {e}")
        return 'no_toxic'


def batch_predict_toxicity(texts: List[str]) -> List[str]:
    """
    Predict toxicity labels for a batch of texts.
    
    Args:
        texts (List[str]): List of texts to analyze
        
    Returns:
        List[str]: List of toxicity labels
    """
    print(f"🤖 Analyzing toxicity for {len(texts)} texts...")
    
    labels = []
    for i, text in enumerate(texts):
        label = predict_toxicity(text)
        labels.append(label)
        
        # Progress indicator for large batches
        if (i + 1) % 100 == 0:
            print(f"   Analyzed {i + 1}/{len(texts)} texts...")
    
    # Summary statistics
    toxic_count = labels.count('toxic')
    non_toxic_count = labels.count('no_toxic')
    
    print(f"✅ Toxicity analysis complete:")
    print(f"   🔴 Toxic: {toxic_count} ({toxic_count/len(labels)*100:.1f}%)")
    print(f"   🟢 Non-toxic: {non_toxic_count} ({non_toxic_count/len(labels)*100:.1f}%)")
    
    return labels


# =============================================================================
# DATASET PREPARATION FUNCTIONS
# =============================================================================

def create_labeled_dataset() -> None:
    """
    Create a labeled dataset from the cleaned merged file.
    
    This function:
    1. Loads the cleaned merged dataset
    2. Applies toxicity labeling to all texts
    3. Creates a balanced dataset if needed
    4. Saves the final labeled dataset
    """
    print("📊 Creating labeled dataset...")
    
    input_file = './data/merge_csv.csv'
    output_file = './data/dataset.csv'
    
    # Check if the cleaned merged file exists
    if not os.path.exists(input_file):
        print(f"❌ Cleaned merged file not found: {input_file}")
        print("💡 Run clean_merged_file() first")
        return
    
    try:
        # Load the cleaned dataset
        print(f"📂 Loading cleaned data from: {input_file}")
        df = pd.read_csv(input_file)
        
        print(f"📊 Dataset size: {len(df)} entries")
        
        # Apply toxicity labeling
        print("🏷️ Applying toxicity labels...")
        df['label'] = batch_predict_toxicity(df['text'].tolist())
        
        # Save the labeled dataset
        df.to_csv(output_file, index=False)
        
        print(f"💾 Labeled dataset saved: {output_file}")
        
        # Display final statistics
        print(f"\n📈 FINAL DATASET STATISTICS:")
        print(f"📊 Total entries: {len(df)}")
        print(f"🔴 Toxic entries: {len(df[df['label'] == 'toxic'])}")
        print(f"🟢 Non-toxic entries: {len(df[df['label'] == 'no_toxic'])}")
        
        # Check for class balance
        toxic_ratio = len(df[df['label'] == 'toxic']) / len(df)
        if toxic_ratio < 0.1 or toxic_ratio > 0.9:
            print("⚠️ Dataset appears imbalanced. Consider collecting more diverse data.")
        else:
            print("✅ Dataset appears reasonably balanced.")
            
    except Exception as e:
        print(f"❌ Error creating labeled dataset: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS (BACKWARD COMPATIBILITY)
# =============================================================================

def merge_file() -> None:
    """Backward compatibility wrapper for merge_csv_files()."""
    merge_csv_files()


def clean_file() -> None:
    """Backward compatibility wrapper for clean_merged_file()."""
    clean_merged_file()


def label(tweet: str) -> str:
    """Backward compatibility wrapper for predict_toxicity()."""
    return predict_toxicity(tweet)


def build_dataset() -> None:
    """Backward compatibility wrapper for create_labeled_dataset()."""
    create_labeled_dataset()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("🛡️ ToxiGuard French Tweets - Helper Functions Test")
    print("="*60)
    
    # Test the main functions
    print("\n🧪 Testing text cleaning...")
    test_text = "Bonjour! Voici un test avec des émojis 😊 et des liens http://example.com #test @user"
    cleaned = clean_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned:  {cleaned}")
    
    print("\n🧪 Testing toxicity detection...")
    test_texts = [
        "Bonjour, comment allez-vous?",
        "C'est une belle journée aujourd'hui.",
        "Je suis très content de ce projet."
    ]
    
    for text in test_texts:
        label = predict_toxicity(text)
        print(f"Text: {text}")
        print(f"Label: {label}")
        print()
    
    print("✅ Helper functions test completed")



