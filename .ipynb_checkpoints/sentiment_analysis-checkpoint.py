from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm # For progress bar
import time
from dotenv import load_dotenv

load_dotenv()

# --- Global Clients ---
VADER_ANALYZER = None
ARE_SENTIMENT_RESOURCES_INITIALIZED = False

def initialize_sentiment_analyzers():
    """Initializes Google NLP client and VADER analyzer."""
    global VADER_ANALYZER, ARE_SENTIMENT_RESOURCES_INITIALIZED
    if ARE_SENTIMENT_RESOURCES_INITIALIZED:
        return True

    print("--- Initializing Sentiment Analyzers ---")
    vader_ok = False

    # Initialize VADER Sentiment Analyzer
    try:
        VADER_ANALYZER = SentimentIntensityAnalyzer()
        print("VADER Sentiment Analyzer initialized.")
        vader_ok = True
    except Exception as e:
        print(f"ERROR: Failed to initialize VADER: {e}")

    ARE_SENTIMENT_RESOURCES_INITIALIZED = vader_ok
    return ARE_SENTIMENT_RESOURCES_INITIALIZED


# --- Function to get sentiment using VADER ---
def get_article_sentiment(text_content):
    """
    Analyzes sentiment of a given text using VADER.
    Returns a dictionary with 'compound', 'pos', 'neu', 'neg' scores, or None on error.
    The 'compound' score is most useful: -1 (most negative) to +1 (most positive).
    """
    if not VADER_ANALYZER:
        print("VADER Analyzer not initialized. Cannot get VADER sentiment.")
        return None
    if not text_content or not isinstance(text_content, str) or len(text_content.strip()) == 0:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0} # Neutral for empty

    try:
        sentiment_scores = VADER_ANALYZER.polarity_scores(text_content)
        return sentiment_scores # Returns dict: {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    except Exception as e:
        print(f"    Error getting VADER sentiment: {e}")
        return None
    
if __name__ == "__main__":
    if not initialize_sentiment_analyzers():
        print("Could not initialize VADER Sentiment Analyzer. Exiting example.")
        exit()

    sample_texts = [
        "This is a wonderfully fantastic and joyous occasion! I am so happy.",
        "I hate this terrible, awful, and disgusting product. It's the worst!",
        "The weather today is neutral and plain.",
        "Despite the initial setbacks, the team showed remarkable resilience and ultimately achieved a surprising victory, though challenges remain.",
        "This is not bad, but it's not good either.",
        "The movie was AMAZING!!! So good, I loved it.",
        "What a horrible experience, I will never come back. Utterly disappointing.",
        "" # Empty text
    ]

    print("\n--- Testing VADER Sentiment via get_article_sentiment ---")
    for i, text in enumerate(sample_texts):
        print(f"\nInput Text {i+1}: \"{text[:70]}{'...' if len(text)>70 else ''}\"")
        sentiment_result = get_article_sentiment(text) 

        if sentiment_result:
            print(f"  Source: {sentiment_result.get('source', 'N/A')}")
            print(f"  Compound: {sentiment_result.get('compound', 0.0):.3f}")
            print(f"  Positive: {sentiment_result.get('pos', 0.0):.3f}")
            print(f"  Neutral:  {sentiment_result.get('neu', 0.0):.3f}")
            print(f"  Negative: {sentiment_result.get('neg', 0.0):.3f}")
        else:
            print("  VADER Sentiment: Error or N/A")

    # Test calling get_article_sentiment again to see if VADER_ANALYZER is reused
    print("\n--- Testing VADER Sentiment again (should reuse initialized analyzer) ---")
    another_text = "This is a second test, it should be quick."
    sentiment_result_2 = get_article_sentiment(another_text)
    if sentiment_result_2:
        print(f"Input Text: \"{another_text}\"")
        print(f"  Compound: {sentiment_result_2.get('compound', 0.0):.3f}")
