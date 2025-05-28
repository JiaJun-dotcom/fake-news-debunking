import pymongo
from pymongo import MongoClient
from google.cloud import language_v1
from google.cloud.language_v1 import types
from tqdm import tqdm # For progress bar
import time
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
CSV_FILE_PATH = "WELFake_Dataset.csv"

# --- Initialize Clients ---
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

language_client = language_v1.LanguageServiceClient()

# --- Process Documents for Sentiment ---
# Fetch documents that don't have sentiment yet (to avoid reprocessing)
documents_to_process = collection.find({"sentiment_score": {"$exists": False}, "text": {"$exists": True, "$ne": None, "$ne": ""}})

doc_list = list(documents_to_process)
print(f"Found {len(doc_list)} documents to process for sentiment analysis.")

batch_size = 50 # Process in batches to manage API calls and updates
for i in tqdm(range(0, len(doc_list), batch_size)):
    batch_docs = doc_list[i:i+batch_size]
    updates = []

    for doc in batch_docs:
        try:
            doc_text = doc.get("text", "")
            if not doc_text or not isinstance(doc_text, str) or len(doc_text.strip()) == 0:
                print(f"Skipping document {doc['_id']} due to empty or invalid text.")
                continue

            max_bytes = 99000 # A bit less than max
            doc_text_bytes = doc_text.encode('utf-8')
            if len(doc_text_bytes) > max_bytes:
                doc_text = doc_text_bytes[:max_bytes].decode('utf-8', 'ignore')


            document = types.Document(
                content=doc_text, type_=types.Document.Type.PLAIN_TEXT
            )
            sentiment = language_client.analyze_sentiment(
                document=document, encoding_type=language_v1.EncodingType.UTF8
            ).document_sentiment

            updates.append({
                "filter": {"_id": doc["_id"]},
                "update": {"$set": {
                    "sentiment_score": sentiment.score,
                    "sentiment_magnitude": sentiment.magnitude
                }}
            })

        except Exception as e:
            print(f"Error processing sentiment for doc {doc['_id']}: {e}")

    # Perform bulk updates to MongoDB
    if updates:
        update_operations = [
            pymongo.UpdateOne(upd["filter"], upd["update"]) for upd in updates
        ]
        try:
            collection.bulk_write(update_operations)
            print(f"Updated sentiment for {len(updates)} documents in batch.")
        except Exception as e:
            print(f"Error bulk writing sentiment updates: {e}")
    time.sleep(1) # Small delay to respect API rate limits if processing very fast

print("Sentiment analysis and MongoDB update complete.")
mongo_client.close()