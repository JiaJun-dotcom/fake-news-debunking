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

MONGO_CLIENT = None
CLASSIFIER_MODEL = None
CLASSIFIER_TOKENIZER = None
VERTEX_EMBEDDING_MODEL = None
NEWS_COLLECTION = None
RESOURCES_INITIALIZED = False

def initialize_classifier_resources():
    global MONGO_CLIENT, NEWS_COLLECTION, CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER, VERTEX_EMBEDDING_MODEL, RESOURCES_INITIALIZED
    
    if RESOURCES_INITIALIZED:
        print("Resources already initialized successfully.")
        return True

    print("--- Initializing Classifier & Vector Search Resources ---")
    mongo_ok = False
    classifier_ok = False
    vertex_embed_ok = False

    # MongoDB Initialization
    if not MONGO_URI:
        print("ERROR: MONGO_URI not found in environment variables.")
    else:
        try:
            MONGO_CLIENT = MongoClient(MONGO_URI)
            db = MONGO_CLIENT[DATABASE_NAME]
            NEWS_COLLECTION = db[COLLECTION_NAME]
            NEWS_COLLECTION.find_one(projection={"_id": 1}) # Test connection
            print(f"MongoDB connection successful to collection: {COLLECTION_NAME}")
            mongo_ok = True
        except Exception as e:
            print(f"ERROR: Could not connect to MongoDB: {e}")
            NEWS_COLLECTION = None 

    # Fine-tuned Classifier Initialization
    if not FINE_TUNED_CLASSIFIER_PATH or not os.path.exists(FINE_TUNED_CLASSIFIER_PATH):
        print(f"ERROR: Fine-tuned classifier model path not found or not set: {FINE_TUNED_CLASSIFIER_PATH}")
    else:
        try:
            CLASSIFIER_MODEL = BertForSequenceClassification.from_pretrained(FINE_TUNED_CLASSIFIER_PATH)
            CLASSIFIER_TOKENIZER = AutoTokenizer.from_pretrained(FINE_TUNED_CLASSIFIER_PATH)
            CLASSIFIER_MODEL.eval()
            print(f"Fine-tuned classifier loaded from: {FINE_TUNED_CLASSIFIER_PATH}")
            classifier_ok = True
        except Exception as e:
            print(f"ERROR: Could not load fine-tuned classifier model: {e}")
            CLASSIFIER_MODEL = None 
            CLASSIFIER_TOKENIZER = None

    # Vertex AI Embedding Model Initialization
    if PROJECT_ID is None or LOCATION is None:
        print("WARNING: GCP_PROJECT_ID or GCP_LOCATION not set. Vertex AI Embedding model cannot initialize.")
    else:
        try:
            print(f"Initializing Vertex AI with Project ID: {PROJECT_ID}, Location: {LOCATION}")
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            print(f"Attempting to load Vertex AI Embedding model: {EMBEDDING_MODEL_NAME}")
            VERTEX_EMBEDDING_MODEL = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
            print(f"Vertex AI Embedding model ({EMBEDDING_MODEL_NAME}) loaded successfully for queries.")
            vertex_embed_ok = True
        except Exception as e:
            print(f"ERROR: Could not load Vertex AI Embedding model: {e}")
            VERTEX_EMBEDDING_MODEL = None # Ensure it's None on failure

    # Determine overall initialization success
    if mongo_ok and vertex_embed_ok: 
        RESOURCES_INITIALIZED = True 
        print("--- Core Resources for Vector Search (MongoDB & Vertex Embeddings) Initialized Successfully ---")
        if classifier_ok:
            print("--- Classifier also initialized successfully ---")
        else:
            print("--- WARNING: Classifier FAILED to initialize. Classification will not work. ---")
    else:
        RESOURCES_INITIALIZED = False
        print("--- CRITICAL WARNING: MongoDB or Vertex Embedding Model FAILED to initialize. Vector search will not work. ---")
        if not classifier_ok:
             print("--- WARNING: Classifier ALSO FAILED to initialize. ---")


    return RESOURCES_INITIALIZED


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
    if NEWS_COLLECTION is None or VERTEX_EMBEDDING_MODEL is None:
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
                "numCandidates": 50, 
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
        return similar_docs
    except Exception as e:
        print(f"  Error during vector search: {e}")
        return []
    
    
if __name__ == "__main__":
    initialize_classifier_resources()

    print("\n--- Testing Semantic Similarity Search (Vector Search) ---")

    test_queries = [
        {
            "title": "LAW ENFORCEMENT ON HIGH ALERT Following Threats Against Cops And Whites On 9-11By #BlackLivesMatter And #FYF911 Terrorists [VIDEO]",
            "text": "No comment is expected from Barack Obama Members of the #FYF911 or #FukYoFlag and #BlackLivesMatter movements called for the lynching and hanging of white people and cops. They encouraged others on a radio show Tuesday night to  turn the tide  and kill white people and cops to send a message about the killing of black people in America.One of the F***YoFlag organizers is called  Sunshine.  She has a radio blog show hosted from Texas called,  Sunshine s F***ing Opinion Radio Show. A snapshot of her #FYF911 @LOLatWhiteFear Twitter page at 9:53 p.m. shows that she was urging supporters to  Call now!! #fyf911 tonight we continue to dismantle the illusion of white Below is a SNAPSHOT Twitter Radio Call Invite   #FYF911The radio show aired at 10:00 p.m. eastern standard time.During the show, callers clearly call for  lynching  and  killing  of white people.A 2:39 minute clip from the radio show can be heard here. It was provided to Breitbart Texas by someone who would like to be referred to as  Hannibal.  He has already received death threats as a result of interrupting #FYF911 conference calls.An unidentified black man said  when those mother f**kers are by themselves, that s when when we should start f***ing them up. Like they do us, when a bunch of them ni**ers takin  one of us out, that s how we should roll up.  He said,  Cause we already roll up in gangs anyway. There should be six or seven black mother f**ckers, see that white person, and then lynch their ass. Let s turn the tables. They conspired that if  cops started losing people,  then  there will be a state of emergency. He speculated that one of two things would happen,  a big-ass [R s?????] war,  or  ni**ers, they are going to start backin  up. We are already getting killed out here so what the f**k we got to lose? Sunshine could be heard saying,  Yep, that s true. That s so f**king true. He said,  We need to turn the tables on them. Our kids are getting shot out here. Somebody needs to become a sacrifice on their side.He said,  Everybody ain t down for that s**t, or whatever, but like I say, everybody has a different position of war.  He continued,  Because they don t give a f**k anyway.  He said again,  We might as well utilized them for that s**t and turn the tables on these n**ers. He said, that way  we can start lookin  like we ain t havin  that many casualties, and there can be more causalities on their side instead of ours. They are out their killing black people, black lives don t matter, that s what those mother f**kers   so we got to make it matter to them. Find a mother f**ker that is alone. Snap his ass, and then f***in hang him from a damn tree. Take a picture of it and then send it to the mother f**kers. We  just need one example,  and  then people will start watchin .  This will turn the tables on s**t, he said. He said this will start  a trickle-down effect.  He said that when one white person is hung and then they are just  flat-hanging,  that will start the  trickle-down effect.  He continued,  Black people are good at starting trends. He said that was how  to get the upper-hand. Another black man spoke up saying they needed to kill  cops that are killing us. The first black male said,  That will be the best method right there. Breitbart Texas previously reported how Sunshine was upset when  racist white people  infiltrated and disrupted one of her conference calls. She subsequently released the phone number of one of the infiltrators. The veteran immediately started receiving threatening calls.One of the #F***YoFlag movement supporters allegedly told a veteran who infiltrated their publicly posted conference call,  We are going to rape and gut your pregnant wife, and your f***ing piece of sh*t unborn creature will be hung from a tree. Breitbart Texas previously encountered Sunshine at a Sandra Bland protest at the Waller County Jail in Texas, where she said all white people should be killed. She told journalists and photographers,  You see this nappy-ass hair on my head?   That means I am one of those more militant Negroes.  She said she was at the protest because  these redneck mother-f**kers murdered Sandra Bland because she had nappy hair like me. #FYF911 black radicals say they will be holding the  imperial powers  that are actually responsible for the terrorist attacks on September 11th accountable on that day, as reported by Breitbart Texas. There are several websites and Twitter handles for the movement. Palmetto Star  describes himself as one of the head organizers. He said in a YouTube video that supporters will be burning their symbols of  the illusion of their superiority,  their  false white supremacy,  like the American flag, the British flag, police uniforms, and Ku Klux Klan hoods.Sierra McGrone or  Nocturnus Libertus  posted,  you too can help a young Afrikan clean their a** with the rag of oppression.  She posted two photos, one that appears to be herself, and a photo of a black man, wiping their naked butts with the American flag.For entire story: Breitbart News"
        },
        {
            "title": "Election Results Disputed Amidst Claims of Irregularities",
            "text": "Following the recent presidential election, the losing candidate has refused to concede, citing widespread voter fraud and calling for a full audit. International observers have expressed concerns over the stability of the region."
        },
        {
            "title": "Tech Giant Launches Revolutionary Smartphone",
            "text": "The latest smartphone from a major tech company boasts an innovative foldable screen, an AI-powered camera system, and unprecedented battery life. Pre-orders have already exceeded expectations."
        },
        { # A query that might be very different from your news data
            "title": "My Cat's Daily Adventures",
            "text": "My cat, Whiskers, spent the morning chasing a sunbeam. Later, he napped on the keyboard. For dinner, he demanded tuna. It was a busy day for him."
        }
    ]

    for i, query in enumerate(test_queries):
        print(f"\n\n--- Query {i+1} ---")
        print(f"Query Title: {query['title']}")
        print(f"Query Text (snippet): {query['text'][:100]}...")

        similar_articles = find_similar_articles_vector_search(query["title"], query["text"], num_results=3)

        if similar_articles:
            print(f"\n  Found {len(similar_articles)} similar articles in MongoDB:")
            for article in similar_articles:
                print(f"    - ID: {article.get('_id')}, Label: {article.get('label')}, Similarity: {article.get('similarity_score', 0.0):.4f}")
                # If you stored 'title_snippet': print(f"      Snippet: {article.get('title_snippet')}")
        else:
            print("  No similar articles found or an error occurred during search.")
        print("-" * 30)