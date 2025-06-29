import os
import re
import json
from dotenv import load_dotenv
import spacy

# --- Load Environment Variables ---
load_dotenv()

LOADED_LANGUAGE_TERMS = {}
CONSPIRACY_MARKERS = []
VAGUE_SOURCES_TERMS = []
SENSATIONAL_PUNCTUATION_PATTERNS = {}
OTHER_CHARGED_PHRASES = {}

LEXICON_FILE_PATH = "./data/lexicons.json"
nlp = spacy.load("en_core_web_sm")

# --- Initialization Function for API Clients & Lexicons ---
import json
# Assuming LANGUAGE_CLIENT, LOADED_LANGUAGE_TERMS, etc. are global as per your code
# Assuming language_v1 is imported and LEXICON_FILE_PATH is defined

# --- Initialization Function for API Clients & Lexicons ---
def initialize_tactic_resources():
    global LOADED_LANGUAGE_TERMS, CONSPIRACY_MARKERS, VAGUE_SOURCES_TERMS, SENSATIONAL_PUNCTUATION_PATTERNS, OTHER_CHARGED_PHRASES

    lexicons_successfully_loaded = False  # Flag for lexicon loading

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
        lexicons_successfully_loaded = True
    except FileNotFoundError:
        print(f"ERROR: Lexicon JSON file not found: {LEXICON_FILE_PATH}. Some tactic detection will be limited.")
        # Decide if this is a critical failure. If so, return False early or keep flag as False.
        # For now, we'll let it proceed to try and init the Google client.
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {LEXICON_FILE_PATH}. Some tactic detection will be limited.")
        # Decide if this is a critical failure.

    # Determine overall success
    if lexicons_successfully_loaded:
        return True
    else:
        print("Tactic resources: Lexicon loading failed or incomplete.")
        return False 

def analyze_text_with_spacy(text_content):
    if not text_content or not isinstance(text_content, str) or len(text_content.strip()) == 0:
        return None
    return nlp(text_content)


# --- Main Tactic Detection Function ---
def detect_disinformation_tactics(title, text, article_url=None):
    """
    Detects various disinformation tactics in a news article.
    Relies on globally initialized LANGUAGE_CLIENT and VADER_ANALYZER.
    """
    detected_tactics = set()
    if not text and not title: return []

    doc = analyze_text_with_spacy(text) if text else None

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

    # c) Urgency (from OTHER_CHARGED_PHRASES in lexicons.json)
    combined_text_lower = ((title.lower() if title else "") + " " + (text.lower() if text else "")).strip()
    if OTHER_CHARGED_PHRASES.get("urgent_phrases"):
        if any(phrase in combined_text_lower for phrase in OTHER_CHARGED_PHRASES["urgent_phrases"]):
            detected_tactics.add("urgency_phrasing")


    # --- 2. Lack of Specificity/Vagueness ---
    # a) Vague Sourcing Phrases (Lexicon)
    if text and any(phrase in text.lower() for phrase in VAGUE_SOURCES_TERMS):
        detected_tactics.add("vague_sourcing_phrases")

    # b) Heuristic for Extremely Low Specific Entity Mentions
    if doc:
        specific_persons = {ent.text for ent in doc.ents if ent.label_ == "PERSON" and len(ent.text.split()) > 1}
        specific_orgs = {ent.text for ent in doc.ents if ent.label_ == "ORG"}
        if len(specific_persons) == 0 and len(specific_orgs) == 0:
            detected_tactics.add("lack_of_specific_source_entities")

    # --- 3. Use of Loaded Language(Negative/Positive Bias) ---
    for category, words in LOADED_LANGUAGE_TERMS.items():
        if any(word in combined_text_lower for word in words):
            detected_tactics.add(f"loaded_language_{category}")

    # --- 4. Conspiracy Markers ---
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

if __name__ == "__main__":
    # load_dotenv() is at the top, so it's already called when script runs.

    initialize_tactic_resources()
    print("\n--- Testing Tactic Detection ---")

    test_cases = [
        {
            "title": "BREAKING!!! ALIENS LAND IN TIMES SQUARE, GOVERNMENT COVER-UP EXPOSED BY INSIDERS!",
            "text": """
            Sources close to the Pentagon, who wish to remain anonymous for their safety, have exclusively revealed
            that an extraterrestrial spacecraft landed in New York's Times Square late last night. "They don't want you to know
            the truth," one insider whispered, "but the evidence is undeniable." Eyewitnesses, described only as "concerned citizens,"
            reported seeing bright lights and strange figures. Mainstream media is silent, proving this is a massive conspiracy.
            Experts say this changes everything. We must wake up! The so-called "official" reports are a complete hoax.
            This shocking development was immediately reported by our brave team. It was seen by many.
            """
        },
        {
            "title": "New Study Suggests Link Between Coffee and Reduced Risk of Certain Diseases",
            "text": """
            A recent epidemiological study published in the "Journal of Internal Medicine" by researchers from Reputable University has indicated a potential
            correlation between regular coffee consumption and a moderately reduced risk of developing type 2 diabetes
            and certain liver conditions. Data from over 100,000 participants were analyzed by these researchers over a ten-year period.
            While the findings are promising, authors caution that correlation does not equal causation and more research is needed.
            "These are interesting preliminary results," stated lead researcher Dr. Eva Rostova, "but people should not drastically
            alter their coffee habits based solely on this study." It is often said by health professionals that moderation is key.
            The research was supported by the National Institutes of Health.
            """
        },
        {
            "title": "URGENT: Stock Market CRASH IMMINENT - Insiders WARN!",
            "text": "Financial gurus are screaming 'SELL EVERYTHING NOW!' A devastating crash is upon us, they say. The elites are preparing, but the mainstream media won't tell you. This is your final warning. It is believed by some that gold will be the only safe haven. The system is rigged!"
        },
        {
            "title": "Local Park Gets New Swingset",
            "text": "The city council announced today that a new swingset has been installed at Willow Creek Park. The installation was completed yesterday. Children were observed enjoying the new equipment this afternoon. The project was funded by local taxes. It is hoped that this will improve the park."
        }
    ]

    for i, case in enumerate(test_cases):
        print(f"\n\n--- Test Case {i+1} ---")
        print(f"Title: {case['title']}")
        print(f"Text (snippet): {case['text'][:150].strip()}...")

        # Call the main tactic detection function
        # source_url is optional for this module if not doing domain checks here
        detected = detect_disinformation_tactics(case["title"], case["text"])

        print(f"\n  Detected Tactics for Case {i+1}:")
        if detected:
            for tactic in detected:
                print(f"    - {tactic}")
        else:
            print("    - No specific tactics flagged by current rules.")
        print("-" * 40)