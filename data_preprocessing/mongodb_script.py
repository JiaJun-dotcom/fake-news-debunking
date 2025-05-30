import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
MONGO_URI = os.environ.get("MONGO_URI")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
CSV_FILE_PATH = "WELFake_Dataset.csv" 

# --- Load CSV ---
try:
    df = pd.read_csv(CSV_FILE_PATH)
    df.rename(columns={
        'Serial number': 'serial_number',
        'Title': 'title',
        'Text': 'text',
        'Label': 'label'
    }, inplace=True)

    df['label'] = df['label'].astype(int)

except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- Connect to MongoDB ---
try:
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# --- Insert Data ---
# Convert DataFrame to list of dictionaries (records)
records = df.to_dict('records')

if records:
    try:
        # Optional: Drop existing collection to avoid duplicates if re-running
        # collection.drop()
        # print(f"Dropped existing collection: {COLLECTION_NAME}")

        collection.insert_many(records)
        print(f"Successfully inserted {len(records)} documents into '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"Error inserting data into MongoDB: {e}")
else:
    print("No records to insert.")

client.close()