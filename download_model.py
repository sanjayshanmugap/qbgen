#!/usr/bin/env python3
"""
Script to download the SentenceTransformer model locally
This avoids Hugging Face rate limits during deployment
"""

import os
from sentence_transformers import SentenceTransformer

def download_model():
    """Download the all-MiniLM-L6-v2 model locally"""
    print("Downloading SentenceTransformer model: all-MiniLM-L6-v2")
    
    # Create models directory
    models_dir = "backend/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Download the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Save the model locally
    model_path = os.path.join(models_dir, "all-MiniLM-L6-v2")
    model.save(model_path)
    
    print(f"Model saved to: {model_path}")
    print("Model download complete!")

if __name__ == "__main__":
    download_model()
