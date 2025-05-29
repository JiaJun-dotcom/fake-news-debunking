from pymongo import MongoClient
from vertexai.language_models import TextEmbeddingModel
import vertexai
from tqdm import tqdm
import time
import pymongo
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI")
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
CSV_FILE_PATH = "WELFake_Dataset.csv"
EMBEDDING_MODEL_NAME = "text-embedding-005"
# There is also "textembedding-gecko-multilingual@001" for multilingual text embedding.

# MAX_CHARS_PER_CHUNK = 18000 * 3

# --- Initialize Clients ---
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
print(f"Using embedding model: {EMBEDDING_MODEL_NAME}")

documents_to_process = collection.find({"text_embedding": {"$exists": False}, "text": {"$exists": True, "$ne": None, "$ne": ""}})
doc_list = list(documents_to_process)
print(f"Found {len(doc_list)} documents to process for embeddings.")

batch_size = 10

for i in tqdm(range(0, len(doc_list), batch_size)):
    current_batch_docs = doc_list[i:i+batch_size]
    texts_to_embed = []
    doc_ids_in_batch = []

    for doc in current_batch_docs:
        content = f"{doc.get('title', '')}\n{doc.get('text', '')}".strip()
        # content = doc.get("text", "").strip() # Or just text
        if content:
            texts_to_embed.append(content)
            doc_ids_in_batch.append(doc["_id"])
        else:
            print(f"Skipping doc {doc['_id']} for embedding due to empty content.")

    if not texts_to_embed:
        continue

    try:
        embeddings = model.get_embeddings(texts_to_embed)
        updates = []
        for idx, embedding_obj in enumerate(embeddings):
            doc_id = doc_ids_in_batch[idx]
            # The embedding values are in embedding_obj.values
            updates.append({
                "filter": {"_id": doc_id},
                "update": {"$set": {"text_embedding": embedding_obj.values}}
            })

        if updates:
            update_operations = [
                pymongo.UpdateOne(upd["filter"], upd["update"]) for upd in updates
            ]
            collection.bulk_write(update_operations)
            # print(f"Generated and stored embeddings for {len(updates)} documents.")

    except Exception as e:
        print(f"Error generating/storing embeddings for batch starting at doc {doc_ids_in_batch[0] if doc_ids_in_batch else 'N/A'}: {e}")
    time.sleep(1) # Respect API rate limits

print("Embedding generation and MongoDB update complete.")
mongo_client.close()