import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax
from vertexai.language_models import TextEmbeddingModel
import vertexai

load_dotenv()

# --- Configuration ---
FINE_TUNED_MODEL_PATH = "./fine_tuned_bertmini"

def initialize_classifier_resources():
    global CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER
    # Fine-tuned DistilBERT Classifier
    try:
        if not os.path.exists(FINE_TUNED_MODEL_PATH):
            print(f"ERROR: Fine-tuned model not found at {FINE_TUNED_MODEL_PATH}. Please train and save it first.")
        CLASSIFIER_MODEL = BertForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
        CLASSIFIER_TOKENIZER = BertTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
        CLASSIFIER_MODEL.eval() # Set to evaluation mode
        print("Fine-tuned BERT classifier loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load fine-tuned DistilBERT model: {e}")

def classify_article_text(title, text_content):
    if not CLASSIFIER_MODEL or not CLASSIFIER_TOKENIZER:
        print("  Classifier model not loaded. Skipping classification.")
        return None

    # Combine title and text for classification for more context
    combined_input = (title if title else "") + " [SEP] " + (text_content if text_content else "")
    if not combined_input.strip() or combined_input.strip() == "[SEP]":
        print("  Empty input for classification.")
        return None

    print("  Classifying article with fine-tuned DistilBERT...")
    inputs = CLASSIFIER_TOKENIZER(combined_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = CLASSIFIER_MODEL(**inputs)
        logits = outputs.logits
        probabilities = softmax(logits, dim=1)

    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class_id].item()

    # Assuming 0 = fake, 1 = real (adjust if your fine-tuning labels were different)
    label_map = {0: "fake", 1: "real"}
    predicted_label = label_map.get(predicted_class_id, "unknown")

    print(f"  Predicted: {predicted_label}, Confidence: {confidence:.4f}")
    return {"prediction": predicted_label, "confidence": confidence}

initialize_classifier_resources()
classify_article_text("trump is dead!!!!!!", "donald trump got shot in california, June 20 2025, and died.")