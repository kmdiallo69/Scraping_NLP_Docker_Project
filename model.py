#!/usr/bin/env python3
"""
ToxiGuard French Tweets - Machine Learning Model Training Module

This module handles the training of the toxicity detection model using SVM
with TF-IDF vectorization. It processes the labeled dataset and creates
production-ready models for the FastAPI service.

Author: ToxiGuard Team
License: MIT
"""

# Standard library imports
import pickle
import os
from typing import Tuple, Optional

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================

# Model storage configuration
MODEL_OUTPUT_PATH = './fastapi/model/'
VECTORIZER_FILENAME = 'vectorizer_tfidf.sav'
MODEL_FILENAME = 'finalized_tfidf.sav'
LABEL_ENCODER_FILENAME = 'label_encoder.sav'

# Dataset configuration
DATASET_PATH = './data/dataset.csv'

# TF-IDF Vectorizer parameters
TFIDF_CONFIG = {
    'ngram_range': (1, 4),  # Use unigrams to 4-grams
    'max_features': 10000,  # Limit vocabulary size for performance
    'min_df': 2,           # Ignore terms that appear in less than 2 documents
    'max_df': 0.95,        # Ignore terms that appear in more than 95% of documents
    'stop_words': None,    # Keep all words for toxicity detection
    'lowercase': True,     # Convert all text to lowercase
    'analyzer': 'word'     # Analyze at word level
}

# SVM GridSearch parameters
SVM_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],           # Regularization parameter
    'gamma': ['scale', 'auto', 0.1, 1],  # Kernel coefficient
    'kernel': ['linear', 'rbf']        # Kernel type
}

# Cross-validation settings
CV_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_model_directory() -> None:
    """
    Create the model output directory if it doesn't exist.
    """
    if not os.path.exists(MODEL_OUTPUT_PATH):
        os.makedirs(MODEL_OUTPUT_PATH)
        print(f"📁 Created model directory: {MODEL_OUTPUT_PATH}")
    else:
        print(f"✅ Model directory exists: {MODEL_OUTPUT_PATH}")


def load_and_validate_dataset(dataset_path: str) -> Optional[pd.DataFrame]:
    """
    Load and validate the labeled dataset.
    
    Args:
        dataset_path (str): Path to the dataset CSV file
        
    Returns:
        Optional[pd.DataFrame]: Loaded dataset or None if invalid
    """
    print(f"📂 Loading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        print("💡 Please run the scraping script first to generate the dataset")
        return None
    
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)
        
        # Validate required columns
        required_columns = ['text', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return None
        
        # Basic data validation
        print(f"📊 Dataset loaded successfully:")
        print(f"   📝 Total samples: {len(df)}")
        print(f"   📄 Text samples: {len(df['text'].dropna())}")
        print(f"   🏷️ Labeled samples: {len(df['label'].dropna())}")
        
        # Check for missing values
        missing_text = df['text'].isna().sum()
        missing_labels = df['label'].isna().sum()
        
        if missing_text > 0:
            print(f"⚠️ Missing text values: {missing_text}")
            df = df.dropna(subset=['text'])
            
        if missing_labels > 0:
            print(f"⚠️ Missing label values: {missing_labels}")
            df = df.dropna(subset=['label'])
        
        # Display label distribution
        label_counts = df['label'].value_counts()
        print(f"\n📈 Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {label}: {count} samples ({percentage:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def prepare_data_for_training(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Prepare the dataset for machine learning training.
    
    Args:
        df (pd.DataFrame): The labeled dataset
        
    Returns:
        Tuple[np.ndarray, np.ndarray, LabelEncoder]: Features, labels, and label encoder
    """
    print("🔄 Preparing data for training...")
    
    # Extract texts and labels
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Encode labels to numeric values
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f"✅ Data preparation complete:")
    print(f"   📝 Text features: {len(texts)}")
    print(f"   🏷️ Encoded labels: {len(encoded_labels)}")
    print(f"   🔢 Label classes: {list(label_encoder.classes_)}")
    
    return np.array(texts), encoded_labels, label_encoder


def create_tfidf_features(texts: np.ndarray) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Create TF-IDF features from text data.
    
    Args:
        texts (np.ndarray): Array of text documents
        
    Returns:
        Tuple[np.ndarray, TfidfVectorizer]: TF-IDF matrix and fitted vectorizer
    """
    print("🔤 Creating TF-IDF features...")
    print(f"   📋 Configuration: {TFIDF_CONFIG}")
    
    # Initialize TF-IDF vectorizer with configuration
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    print(f"✅ TF-IDF vectorization complete:")
    print(f"   📊 Feature matrix shape: {tfidf_matrix.shape}")
    print(f"   📝 Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"   💾 Memory usage: {tfidf_matrix.data.nbytes / 1024 / 1024:.2f} MB")
    
    return tfidf_matrix, vectorizer


def train_svm_model(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """
    Train an SVM model with hyperparameter optimization.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        
    Returns:
        SVC: Trained SVM model
    """
    print("🧠 Training SVM model with hyperparameter optimization...")
    print(f"   📊 Training data shape: {X_train.shape}")
    print(f"   🔍 Parameter grid: {SVM_PARAM_GRID}")
    print(f"   🔄 Cross-validation folds: {CV_FOLDS}")
    
    # Initialize SVM classifier
    svm = SVC(random_state=RANDOM_STATE)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=SVM_PARAM_GRID,
        cv=CV_FOLDS,
        scoring='f1_weighted',  # Use weighted F1 for imbalanced data
        verbose=2,             # Show progress
        n_jobs=-1             # Use all available cores
    )
    
    # Fit the model
    print("🚀 Starting grid search training...")
    grid_search.fit(X_train, y_train)
    
    # Display results
    print(f"✅ Training complete!")
    print(f"   🏆 Best parameters: {grid_search.best_params_}")
    print(f"   📊 Best CV score: {grid_search.best_score_:.4f}")
    print(f"   🔧 Best estimator: {grid_search.best_estimator_}")
    
    return grid_search.best_estimator_


def evaluate_model(model: SVC, X_test: np.ndarray, y_test: np.ndarray, 
                   label_encoder: LabelEncoder) -> None:
    """
    Evaluate the trained model on test data.
    
    Args:
        model (SVC): Trained SVM model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        label_encoder (LabelEncoder): Label encoder for class names
    """
    print("📊 Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📈 MODEL PERFORMANCE METRICS:")
    print(f"{'='*50}")
    print(f"🎯 Accuracy:  {accuracy:.4f}")
    print(f"🎨 Precision: {precision:.4f}")
    print(f"🔍 Recall:    {recall:.4f}")
    print(f"⚖️  F1-Score:  {f1:.4f}")
    
    # Detailed classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(f"{'='*50}")
    target_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    
    # Confusion matrix
    print(f"\n🔀 CONFUSION MATRIX:")
    print(f"{'='*50}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Classes: {target_names}")
    print(cm)


def save_model_artifacts(model: SVC, vectorizer: TfidfVectorizer, 
                        label_encoder: LabelEncoder) -> None:
    """
    Save all model artifacts for production deployment.
    
    Args:
        model (SVC): Trained SVM model
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer
        label_encoder (LabelEncoder): Fitted label encoder
    """
    print("💾 Saving model artifacts...")
    
    # Ensure output directory exists
    ensure_model_directory()
    
    try:
        # Save the trained model
        model_path = os.path.join(MODEL_OUTPUT_PATH, MODEL_FILENAME)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Model saved: {model_path}")
        
        # Save the vectorizer
        vectorizer_path = os.path.join(MODEL_OUTPUT_PATH, VECTORIZER_FILENAME)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"✅ Vectorizer saved: {vectorizer_path}")
        
        # Save the label encoder
        encoder_path = os.path.join(MODEL_OUTPUT_PATH, LABEL_ENCODER_FILENAME)
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"✅ Label encoder saved: {encoder_path}")
        
        print(f"\n🎉 All model artifacts saved successfully!")
        print(f"📁 Model directory: {MODEL_OUTPUT_PATH}")
        
    except Exception as e:
        print(f"❌ Error saving model artifacts: {e}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def build_model() -> None:
    """
    Main function to build and train the toxicity detection model.
    
    This function orchestrates the entire model training pipeline:
    1. Load and validate the dataset
    2. Prepare data for training
    3. Create TF-IDF features
    4. Split data into training and testing sets
    5. Train SVM model with hyperparameter optimization
    6. Evaluate model performance
    7. Save model artifacts for production
    """
    print("🛡️ ToxiGuard French Tweets - Model Training Pipeline")
    print("="*70)
    print("🎯 This pipeline will:")
    print("   1. Load and validate the labeled dataset")
    print("   2. Create TF-IDF features from text data")
    print("   3. Train an optimized SVM classifier")
    print("   4. Evaluate model performance")
    print("   5. Save production-ready model artifacts")
    print("="*70)
    
    try:
        # Step 1: Load and validate dataset
        print(f"\n📋 STEP 1: DATASET LOADING")
        df = load_and_validate_dataset(DATASET_PATH)
        if df is None:
            return
        
        # Step 2: Prepare data for training
        print(f"\n📋 STEP 2: DATA PREPARATION")
        texts, labels, label_encoder = prepare_data_for_training(df)
        
        # Step 3: Create TF-IDF features
        print(f"\n📋 STEP 3: FEATURE EXTRACTION")
        tfidf_matrix, vectorizer = create_tfidf_features(texts)
        
        # Step 4: Split data
        print(f"\n📋 STEP 4: TRAIN/TEST SPLIT")
        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_matrix, labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels  # Maintain class distribution
        )
        
        print(f"✅ Data split complete:")
        print(f"   🏋️ Training samples: {X_train.shape[0]}")
        print(f"   🧪 Testing samples: {X_test.shape[0]}")
        print(f"   📊 Feature dimensions: {X_train.shape[1]}")
        
        # Step 5: Train model
        print(f"\n📋 STEP 5: MODEL TRAINING")
        model = train_svm_model(X_train, y_train)
        
        # Step 6: Evaluate model
        print(f"\n📋 STEP 6: MODEL EVALUATION")
        evaluate_model(model, X_test, y_test, label_encoder)
        
        # Step 7: Save model artifacts
        print(f"\n📋 STEP 7: SAVING MODEL ARTIFACTS")
        save_model_artifacts(model, vectorizer, label_encoder)
        
        # Success message
        print(f"\n🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("📁 Generated files:")
        print(f"   🤖 {MODEL_OUTPUT_PATH}{MODEL_FILENAME} - Trained SVM model")
        print(f"   🔤 {MODEL_OUTPUT_PATH}{VECTORIZER_FILENAME} - TF-IDF vectorizer")
        print(f"   🏷️ {MODEL_OUTPUT_PATH}{LABEL_ENCODER_FILENAME} - Label encoder")
        print("\n🚀 You can now start the FastAPI service!")
        
    except Exception as e:
        print(f"\n❌ Error during model training: {e}")
        print("💡 Please check the error message and try again")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    build_model()


