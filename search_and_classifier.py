import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.nn.functional import softmax
from vertexai.language_models import TextEmbeddingModel
import vertexai

load_dotenv()

# ** Calls fine-tuned text classifier to output whether it is fake/real news, coupled with semantic similarity search with MongoDB to check against database of how similar it is to [? amount] FAKE/REAL news in vector db.

# --- Configuration ---
MONGO_URI = os.environ.get("MONGO_URI")
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
EMBEDDING_MODEL_NAME = "text-embedding-005"
FINE_TUNED_MODEL_PATH = "./fine_tuned_bertmini"
VECTOR_SEARCH_INDEX_NAME = "vector_index"

def initialize_classifier_resources():
    global MONGO_CLIENT, NEWS_COLLECTION, CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER, VERTEX_EMBEDDING_MODEL

    # MongoDB
    if not MONGO_URI:
        print("ERROR: MONGO_URI not found in environment variables.")
        return False
    try:
        MONGO_CLIENT = MongoClient(MONGO_URI)
        db = MONGO_CLIENT[DATABASE_NAME]
        NEWS_COLLECTION = db[COLLECTION_NAME]
        # Test connection
        NEWS_COLLECTION.find_one(projection={"_id": 1}) 
        print("MongoDB connection successful.")
    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB: {e}")
        return False

    # Fine-tuned BERT Classifier
    try:
        if not os.path.exists(FINE_TUNED_MODEL_PATH):
            print(f"ERROR: Fine-tuned model not found at {FINE_TUNED_MODEL_PATH}. Please train and save it first.")
            return False
        CLASSIFIER_MODEL = BertForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
        CLASSIFIER_TOKENIZER = BertTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
        CLASSIFIER_MODEL.eval() # Set to evaluation mode
        print("Fine-tuned BERT classifier loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load fine-tuned BERT model: {e}")
        return False

    # Vertex AI Embedding Model (for querying vector search)
    if not PROJECT_ID:
        print("WARNING: PROJECT_ID not set. Vertex AI Embedding model for vector search query might not initialize.")
    else:
        try:
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            try:
                VERTEX_EMBEDDING_MODEL = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
                print(f"Vertex AI Embedding model ({EMBEDDING_MODEL_NAME}) loaded successfully for queries.")
        except Exception as e:
            print(f"ERROR: Could not load Vertex AI Embedding model: {e}")
    return True


# --- 2. Inference with Fine-tuned BERT ---
def classify_article_text(title, text_content):
    if not CLASSIFIER_MODEL or not CLASSIFIER_TOKENIZER:
        print("  Classifier model not loaded. Skipping classification.")
        return None

    # Combine title and text for classification for more context
    combined_input = (title if title else "") + " [SEP] " + (text_content if text_content else "")
    if not combined_input.strip() or combined_input.strip() == "[SEP]":
        print("  Empty input for classification.")
        return None

    print("  Classifying article with fine-tuned BERT...")
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

# --- 3. Semantic Similarity Search (Atlas Vector Search) ---
def find_similar_articles_vector_search(query_title, query_text, num_results=5):
    if not NEWS_COLLECTION or not VERTEX_EMBEDDING_MODEL:
        print("  MongoDB connection or Vertex Embedding model not available. Skipping vector search.")
        return []

    print(f"  Performing vector search for articles similar to: {query_title[:50]}...")
    # Combine title and text for the query embedding
    query_content_for_embedding = (query_title if query_title else "") + "\n" + (query_text if query_text else "")
    if not query_content_for_embedding.strip():
        print("  Empty content for vector search query.")
        return []

    try:
        embedding_response = VERTEX_EMBEDDING_MODEL.get_embeddings([query_content_for_embedding])
        query_vector = embedding_response[0].values
    except Exception as e:
        print(f"  Error generating embedding for vector search query: {e}")
        return []

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "text_embedding",
                "queryVector": query_vector,
                "numCandidates": 10, # Adjust as needed
                "limit": num_results
            }
        },
        {
            "$project": {
                "_id": 0,
                "title": 1,
                "label": 1, 
                "sentiment_score": {"$ifNull": ["$sentiment_score", "N/A"]}, # Handle missing sentiment
                "similarity_score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    try:
        similar_docs = list(NEWS_COLLECTION.aggregate(pipeline))
        print(f"  Found {len(similar_docs)} similar articles via vector search.")
        return similar_docs
    except Exception as e:
        print(f"  Error during vector search: {e}")
        return []
    
    
