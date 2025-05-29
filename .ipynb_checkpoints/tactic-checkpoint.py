import os
import re
import json
from dotenv import load_dotenv
from google.cloud import language_v1
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # For VADER

# --- Load Environment Variables ---
load_dotenv()

# --- Global Clients & Lexicons (Initialize once) ---
LANGUAGE_CLIENT = None
VADER_ANALYZER = None # For VADER

LOADED_LANGUAGE_TERMS = {}
CONSPIRACY_MARKERS = []
VAGUE_SOURCES_TERMS = []
SENSATIONAL_PUNCTUATION_PATTERNS = {}
OTHER_CHARGED_PHRASES = {}

LEXICON_FILE_PATH = "lexicons.json"

# --- Initialization Function for API Clients & Lexicons ---
def initialize_resources():
    global LANGUAGE_CLIENT, VADER_ANALYZER
    global LOADED_LANGUAGE_TERMS, CONSPIRACY_MARKERS, VAGUE_SOURCES_TERMS, SENSATIONAL_PUNCTUATION_PATTERNS, OTHER_CHARGED_PHRASES

    # Load Lexicons
    try:
        with open(LEXICON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        LOADED_LANGUAGE_TERMS = data.get("loaded_language_terms", {})
        CONSPIRACY_MARKERS = data.get("conspiracy_markers", [])
        VAGUE_SOURCES_TERMS = data.get("vague_sources_terms", [])
        SENSATIONAL_PUNCTUATION_PATTERNS = data.get("sensational_punctuation_patterns", {})
        OTHER_CHARGED_PHRASES = data.get("other_charged_phrases", {})
        print(f"Lexicons loaded successfully from {LEXICON_FILE_PATH}.")
    except FileNotFoundError:
        print(f"ERROR: Lexicon JSON file not found: {LEXICON_FILE_PATH}. Some tactic detection will be limited.")
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {LEXICON_FILE_PATH}. Some tactic detection will be limited.")

    # Initialize Google Natural Language API Client
    try:
        LANGUAGE_CLIENT = language_v1.LanguageServiceClient()
        print("Google Natural Language API client initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Google Natural Language API client: {e}")

    # Initialize VADER Sentiment Analyzer
    try:
        VADER_ANALYZER = SentimentIntensityAnalyzer()
        print("VADER Sentiment Analyzer initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize VADER: {e}")


# --- Helper Function for Google NLP API (Sentiment, Entities, Syntax) ---
def analyze_text_with_google_nlp_full(text_content):
    """Analyzes text for sentiment, entities, and syntax using Google NLP API."""
    if not LANGUAGE_CLIENT or not text_content or not isinstance(text_content, str) or len(text_content.strip()) == 0:
        return None, None, None
    try:
        document = language_v1.types.Document(content=text_content, type_=language_v1.types.Document.Type.PLAIN_TEXT)
        features = language_v1.AnnotateTextRequest.Features(
            extract_syntax=True,
            extract_entities=True,
            extract_document_sentiment=True
        )
        response = LANGUAGE_CLIENT.annotate_text(document=document, features=features, encoding_type=language_v1.EncodingType.UTF8)
        return response.document_sentiment, response.entities, response.sentences, response.tokens
    except Exception as e:
        print(f"Error in Google NLP API (annotate_text) call: {e}")
        return None, None, None, None


# --- Main Tactic Detection Function ---
def detect_disinformation_tactics(title, text):
    """
    Detects various disinformation tactics in a news article.
    Relies on globally initialized LANGUAGE_CLIENT and VADER_ANALYZER.
    """
    detected_tactics = set()
    if not text and not title: return []

    # --- A. Initial Full NLP Analysis (Sentiment, Entities, Syntax from Google) ---
    gcp_sentiment, gcp_entities, gcp_sentences, gcp_tokens = None, None, None, None
    if text: # Only analyze text if it exists
        gcp_sentiment, gcp_entities, gcp_sentences, gcp_tokens = analyze_text_with_google_nlp_full(text)

    # --- 1. Sensationalism/Clickbait ---
    # a) Excessive Capitalization in Title
    if title and len(title.replace(" ", "")) > 0: # Avoid division by zero for empty titles
        if sum(1 for char in title if char.isupper()) / len(title.replace(" ", "")) > 0.6 and len(title) > 10:
            detected_tactics.add("sensational_title_excessive_caps")

    # b) Excessive Punctuation (from lexicons.json)
    if title:
        if SENSATIONAL_PUNCTUATION_PATTERNS.get("title_exclamations") and \
           re.search(SENSATIONAL_PUNCTUATION_PATTERNS["title_exclamations"], title):
            detected_tactics.add("sensational_title_exclamations")
        if SENSATIONAL_PUNCTUATION_PATTERNS.get("title_questions") and \
           re.search(SENSATIONAL_PUNCTUATION_PATTERNS["title_questions"], title):
            detected_tactics.add("sensational_title_questions")

    if text:
        num_words_text = len(text.split())
        if num_words_text > 0: # Avoid division by zero
            exclamation_density_threshold = SENSATIONAL_PUNCTUATION_PATTERNS.get("text_exclamations_density_threshold", 0.5)
            if text.count('!') / (num_words_text / 100.0 + 1e-6) > exclamation_density_threshold:
                detected_tactics.add("sensational_text_exclamations")

    # c) Emotionally Charged Words (using VADER for intensity)
    if VADER_ANALYZER and text:
        vs = VADER_ANALYZER.polarity_scores(text)
        # VADER's compound score is a good indicator of overall sentiment intensity
        if vs['compound'] >= 0.75: # Strongly positive
            detected_tactics.add("highly_positive_text")
        elif vs['compound'] <= -0.75: # Strongly negative
            detected_tactics.add("highly_negative_text")
        
        # Check for words with high individual intensity from VADER's lexicon (more granular)
        # This requires accessing VADER's internal lexicon or pre-processing it.
        # For simplicity, we'll rely on the compound score and GCP sentiment for now.
        # A more advanced VADER usage would iterate through words and check their lexicon scores.

    # d) Urgency (from OTHER_CHARGED_PHRASES in lexicons.json)
    combined_text_lower = ((title.lower() if title else "") + " " + (text.lower() if text else "")).strip()
    if OTHER_CHARGED_PHRASES.get("urgent_phrases"):
        if any(phrase in combined_text_lower for phrase in OTHER_CHARGED_PHRASES["urgent_phrases"]):
            detected_tactics.add("urgency_phrasing")
            
    # e) General high emotionality from GCP Sentiment
    if gcp_sentiment and (gcp_sentiment.score > 0.7 or gcp_sentiment.score < -0.7):
        detected_tactics.add("highly_emotional_sentiment")


    # --- 2. Lack of Specificity/Vagueness ---
    # a) Vague Sourcing Phrases (Lexicon)
    if text and any(phrase in text.lower() for phrase in VAGUE_SOURCES_TERMS):
        detected_tactics.add("vague_sourcing_phrases")

    # b) Heuristic for Extremely Low Specific Entity Mentions
    if gcp_entities and text:
        specific_persons = {
            e.name for e in gcp_entities
            if e.type_ == language_v1.types.Entity.Type.PERSON and
               len(e.name.split()) > 1 and e.salience > 0.015 # Slightly higher salience
        }
        specific_orgs = {
            e.name for e in gcp_entities
            if e.type_ == language_v1.types.Entity.Type.ORGANIZATION and
               not any(vs.lower() in e.name.lower() for vs in VAGUE_SOURCES_TERMS) and
               e.salience > 0.015
        }
        num_words = len(text.split())
        if num_words > 250:
            if len(specific_persons) == 0 and len(specific_orgs) == 0:
                detected_tactics.add("lack_of_specific_source_entities")

    # --- 3. Appeals to Emotion over Logic ---
    if gcp_sentiment and text:
        num_words = len(text.split())
        # High magnitude relative to text length
        if gcp_sentiment.magnitude > (num_words / 100.0 * 1.8) and num_words > 70:
            numbers_in_text = re.findall(r'\b\d{2,}\b', text)
            location_entities = [e.name for e in gcp_entities if e.type_ == language_v1.types.Entity.Type.LOCATION and e.salience > 0.01] if gcp_entities else []
            event_entities = [e.name for e in gcp_entities if e.type_ == language_v1.types.Entity.Type.EVENT and e.salience > 0.01] if gcp_entities else []

            num_specific_actors = 0
            if 'specific_persons' in locals() and 'specific_orgs' in locals(): 
                 num_specific_actors = len(specific_persons) + len(specific_orgs)
            elif gcp_entities:
                 num_specific_actors = len([e for e in gcp_entities if (e.type_ == language_v1.types.Entity.Type.PERSON or e.type_ == language_v1.types.Entity.Type.ORGANIZATION) and e.salience > 0.015])


            factual_marker_count = len(numbers_in_text) + len(location_entities) + len(event_entities) + num_specific_actors
            if factual_marker_count < (num_words / 100.0): # e.g., less than 1 factual marker per 100 words
                detected_tactics.add("high_appeal_to_emotion_with_low_factual_markers")

    # --- 4. Use of Loaded Language(Negative/Positive Bias) ---
    for category, words in LOADED_LANGUAGE_TERMS.items():
        if any(word in combined_text_lower for word in words):
            detected_tactics.add(f"loaded_language_{category}")

    # --- 5. Conspiracy Markers ---
    if text and any(phrase in combined_text_lower for phrase in CONSPIRACY_MARKERS):
        detected_tactics.add("conspiracy_markers_present")

#     # --- 6. Syntax Analysis (e.g., Excessive Passive Voice) ---
#     if gcp_tokens and gcp_sentences: # Check if syntax analysis results are available
#         passive_voice_count = 0
#         total_verbs = 0
#         for token in gcp_tokens:
#             if token.part_of_speech.tag == language_v1.PartOfSpeech.Tag.VERB:
#                 total_verbs += 1
#                 # Check for passive voice: typically auxiliary verb (like 'be', 'get') + past participle
#                 # This is a simplified check. True passive voice detection is more complex.
#                 # Google NLP's dependency parse (token.dependency_edge) is more robust for this.
#                 # Example: if token.dependency_edge.label == language_v1.DependencyEdge.Label.PASSIVE_AUXILIARY
#                 # For a simpler heuristic here:
#                 if token.dependency_edge.label == language_v1.DependencyEdge.Label.AUX_PASS: # AUX_PASS is often used for passive auxiliaries
#                      passive_voice_count +=1
#                 elif token.part_of_speech.form == language_v1.PartOfSpeech.Form.PAST_PARTICIPLE and \
#                      token.dependency_edge.label in [language_v1.DependencyEdge.Label.AGENT, language_v1.DependencyEdge.Label.PATIENT]: # Heuristic
#                      # This part is tricky and might need more refinement based on dependency tree structure
#                      pass # More complex logic needed here for reliable passive detection from just POS and basic dep_edge

#         # A more direct check using dependency labels if available and understood
#         # This is a placeholder for a more robust passive voice check using the full dependency tree.
#         # For now, let's use a simpler count of AUX_PASS as a proxy.
#         # A more robust way: iterate sentences, then tokens, build dependency tree or look for specific patterns.
#         # For this example, we'll use the count of AUX_PASS from tokens.
        
#         # Count passive auxiliaries directly from tokens
#         passive_aux_count = sum(1 for token in gcp_tokens if token.dependency_edge.label == language_v1.DependencyEdge.Label.AUX_PASS)

#         if total_verbs > 5 and passive_aux_count > 0: # Ensure there are some verbs to compare against
#             passive_ratio = passive_aux_count / float(total_verbs)
#             if passive_ratio > 0.25: # If more than 25% of verbs are involved in passive constructions (heuristic)
#                 detected_tactics.add("excessive_passive_voice_heuristic")
#         elif total_verbs <=5 and passive_aux_count > 1: # If few verbs but more than one is passive
#              detected_tactics.add("high_passive_voice_in_short_text_heuristic")


    return list(detected_tactics)