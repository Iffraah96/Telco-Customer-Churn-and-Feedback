import os
import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Initialize spaCy English model for text preprocessing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text_string):
    """ Cleans text by lowercasing, removing punctuation/numbers, 

    and extracting lemmas using spaCy. """
    if not isinstance(text_string, str):
        return ""
    doc = nlp(text_string.lower())
    clean_tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and not token.like_num and token.text.strip()
    ]
    return " ".join(clean_tokens)

def run_pipeline():
    print("=== STARTING HYBRID CHURN ML PIPELINE ===\n")
    
    # 1. Load Data
    data_path = 'data/telco_churn_feedback.csv'
    if not os.path.exists(data_path):
        # Fallback if data folder isn't used yet
        data_path = 'telco_churn_feedback.csv'
        
    print(f"[1/6] Loading raw dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    # 2. Preprocess Tabular Fields
    print("[2/6] Cleaning tabular columns...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Define columns to isolate for the clean tabular representation
    ignore_cols = ['customerID', 'Churn', 'CustomerFeedback', 'PromptInput', 'Cleaned_Feedback']
    X_pure_tabular = df.drop(columns=ignore_cols, errors='ignore')
    
    # One-Hot Encode legacy category data
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [col for col in X_pure_tabular.columns if col not in num_cols]
    X_tabular_encoded = pd.get_dummies(X_pure_tabular, columns=cat_cols, drop_first=True)
    
    # 3. Text Preprocessing via NLP Engine
    print("[3/6] Running text through spaCy NLP engine (This might take a minute)...")
    df['Cleaned_Feedback'] = df['CustomerFeedback'].fillna("").apply(clean_text)
    
    # 4. TF-IDF Vectorization with Data Leakage Mitigation
    print("[4/6] Vectorizing text and applying custom stop-words to eliminate leakage...")
    leakage_words = ['churn', 'churned', 'cancel', 'canceled', 'cancelling', 'cancellation', 'leave', 'left', 'yes', 'no']
    custom_stop_words = list(text.ENGLISH_STOP_WORDS.union(leakage_words))
    
    tfidf = TfidfVectorizer(max_features=500, stop_words=custom_stop_words)
    X_tfidf = tfidf.fit_transform(df['Cleaned_Feedback']).toarray()
    tfidf_df = pd.DataFrame(X_tfidf, columns=tfidf.get_feature_names_out())
    
    # 5. Feature Concatenation (The Hybrid Build)
    print("[5/6] Merging tabular features and NLP features side-by-side...")
    X_tabular_encoded = X_tabular_encoded.reset_index(drop=True)
    tfidf_df = tfidf_df.reset_index(drop=True)
    X_hybrid = pd.concat([X_tabular_encoded, tfidf_df], axis=1)
    
    print(f"      -> Final Hybrid Matrix Shape: {X_hybrid.shape}")
    
    # 6. Train-Test Splits and Machine Learning
    print("[6/6] Splitting datasets and training models...")
    
    # Split for baseline
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_tabular_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    baseline_model = RandomForestClassifier(random_state=42)
    baseline_model.fit(X_train_b, y_train_b)
    y_pred_b = baseline_model.predict(X_test_b)
    
    # Split for hybrid
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_hybrid, y, test_size=0.2, random_state=42, stratify=y
    )
    hybrid_model = RandomForestClassifier(random_state=42)
    hybrid_model.fit(X_train_h, y_train_h)
    y_pred_h = hybrid_model.predict(X_test_h)
    
    # --- EVALUATION PRINTOUTS ---
    print("\n" + "="*40)
    print("         EVALUATION REPORTS")
    print("="*40)
    print("\n--- TRUE TABULAR BASELINE PERFORMANCE ---")
    print(classification_report(y_test_b, y_pred_b))
    
    print("\n--- CLEAN HYBRID MODEL PERFORMANCE ---")
    print(classification_report(y_test_h, y_pred_h))
    
    # Inspect Top Features
    importances = hybrid_model.feature_importances_
    feature_names = X_hybrid.columns
    indices = np.argsort(importances)[::-1]
    
    print("\n--- TOP 5 DOMINANT DISCRIMINATING FEATURES ---")
    for f in range(5):
        print(f" {f+1}. Feature: {feature_names[indices[f]]:<20} Weight: {importances[indices[f]]:.4f}")
    print("\n=== PIPELINE EXECUTION SUCCESSFUL ===")

if __name__ == "__main__":
    run_pipeline()
