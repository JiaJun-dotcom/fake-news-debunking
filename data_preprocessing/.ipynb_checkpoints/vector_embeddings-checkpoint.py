from pymongo import MongoClient
from vertexai.language_models import TextEmbeddingModel
import vertexai
from tqdm import tqdm
import time
import pymongo
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI")
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
CSV_FILE_PATH = "../data/WELFake_Dataset.csv"
EMBEDDING_MODEL_NAME = "text-embedding-005"
# There is also "textembedding-gecko-multilingual@001" for multilingual text embedding.

API_CALL_BATCH_SIZE = 10 

# --- Initialize Clients ---
print("Initializing clients...")
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    # Test connection
    collection.estimated_document_count()
    print(f"MongoDB connection successful. Using DB: '{DATABASE_NAME}', Collection: '{COLLECTION_NAME}'")
except Exception as e:
    print(f"ERROR: Could not connect to MongoDB: {e}")
    exit()

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
print(f"Using embedding model: {EMBEDDING_MODEL_NAME}")
# --- Load CSV Data ---
print(f"Loading CSV data from: {CSV_FILE_PATH}")
try:
    df = pd.read_csv(CSV_FILE_PATH)
    # Standardize column names
    column_map = {}
    if 'Serial number' in df.columns: column_map['Serial number'] = '_id'
    elif 'Unnamed: 0' in df.columns: column_map['Unnamed: 0'] = '_id' # Common if index was saved from pandas
    if 'title' in df.columns: column_map['title'] = 'title_orig'
    if 'text' in df.columns: column_map['text'] = 'text_orig'
    if 'label' in df.columns: column_map['label'] = 'label'
    
    df.rename(columns=column_map, inplace=True)

    if '_id' not in df.columns:
        raise ValueError("CSV must have a 'Serial number' or 'Unnamed: 0' column to use as _id.")
    if 'text_orig' not in df.columns:
        raise ValueError("CSV must have a 'Text' column (renamed to 'text_orig').")
    if 'label' not in df.columns:
        raise ValueError("CSV must have a 'Label' column (renamed to 'label').")

    df['label'] = df['label'].astype(int)
    df['title_orig'] = df['title_orig'].fillna("")
    df['text_orig'] = df['text_orig'].fillna("")
    print(f"Loaded {len(df)} rows from CSV.")
except FileNotFoundError:
    print(f"ERROR: CSV file not found at {CSV_FILE_PATH}")
    exit()
except Exception as e:
    print(f"ERROR loading or processing CSV: {e}")
    exit()

# --- Prepare list of documents from CSV that need embedding ---
docs_to_process_from_csv = []
print("Checking which documents from CSV need to be processed (embedding not in MongoDB)...")
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Pre-checking CSV against DB"):
    doc_id_csv = row["_id"]
    # Check if this document (by _id) already has an embedding in MongoDB
    if collection.count_documents({"_id": doc_id_csv, "text_embedding": {"$exists": True}}) > 0:
        continue # Skip if already processed and embedded in DB

    title_for_embedding = row.get("title_orig", "")
    text_for_embedding = row.get("text_orig", "")
    label = int(row["label"])
    content_to_embed = f"{title_for_embedding}\n{text_for_embedding}".strip()

    if not content_to_embed:
        print(f"Skipping document with _id {doc_id_csv} from CSV due to empty content.")
        continue
    
    docs_to_process_from_csv.append({
        "doc_id": doc_id_csv,
        "text_to_embed": content_to_embed,
        "label": label
    })

if not docs_to_process_from_csv:
    print("No new documents from CSV require embedding based on current MongoDB state.")
    mongo_client.close()
    exit()

print(f"Found {len(docs_to_process_from_csv)} documents from CSV to process for embeddings.")

# For batching MongoDB writes
mongo_write_operations = []
MONGO_WRITE_BATCH_SIZE = 100 # How many docs to prepare before writing to MongoDB

# Process in batches for the embedding API
for i in tqdm(range(0, len(docs_to_process_from_csv), API_CALL_BATCH_SIZE), desc="Embedding API Batches"):
    current_api_batch_data = docs_to_process_from_csv[i : i + API_CALL_BATCH_SIZE]
    
    texts_for_api = [item["text_to_embed"] for item in current_api_batch_data]
    doc_ids_in_api_batch = [item["doc_id"] for item in current_api_batch_data]
    labels_in_api_batch = [item["label"] for item in current_api_batch_data]

    if not texts_for_api:
        continue

    try:
        embeddings_result = model.get_embeddings(texts_for_api) # This is the API call
        
        for idx, embedding_obj in enumerate(embeddings_result):
            doc_id = doc_ids_in_api_batch[idx]
            label = labels_in_api_batch[idx]
            embedding_vector = embedding_obj.values

            # Prepare lean document for MongoDB
            document_for_mongo = {
                "_id": doc_id, # Use the ID from CSV
                "label": label,
                "text_embedding": embedding_vector
            }
            # Add to batch for MongoDB write
            # Using UpdateOne with upsert=True will insert if _id doesn't exist, or update if it does.
            # This is robust if the script is run multiple times or if some pre-check failed.
            mongo_write_operations.append(
                pymongo.UpdateOne({"_id": doc_id}, {"$set": document_for_mongo}, upsert=True)
            )

    except Exception as e:
        print(f"ERROR generating embeddings for API batch starting with doc_id {doc_ids_in_api_batch[0] if doc_ids_in_api_batch else 'N/A'}: {e}")
        print(f"  This batch will be skipped. Problematic texts (first 100 chars):")
        for txt_idx, problem_text in enumerate(texts_for_api):
             print(f"    Doc ID {doc_ids_in_api_batch[txt_idx]}: {problem_text[:100]}...")
        # Continue to the next API batch, skipping the current failed one
        continue 
    
    # Write to MongoDB in batches (can be less frequent than API calls if desired)
    if len(mongo_write_operations) >= MONGO_WRITE_BATCH_SIZE:
        try:
            collection.bulk_write(mongo_write_operations)
            # print(f"  Successfully wrote {len(mongo_write_operations)} documents to MongoDB.")
            mongo_write_operations = [] # Reset batch
        except Exception as e:
            print(f"  ERROR bulk writing to MongoDB: {e}")
            mongo_write_operations = [] # Clear batch even on error to avoid reprocessing same failed data
        time.sleep(0.2) # Small delay after a MongoDB bulk write

    time.sleep(1) # Respect API rate limits between batches of get_embeddings calls

# Write any remaining MongoDB operations after the loop
if mongo_write_operations:
    try:
        collection.bulk_write(mongo_write_operations)
        print(f"  Successfully wrote final {len(mongo_write_operations)} documents to MongoDB.")
    except Exception as e:
        print(f"  ERROR bulk writing final batch to MongoDB: {e}")

print("\nEmbedding generation from CSV and MongoDB lean insertion complete.")
mongo_client.close()