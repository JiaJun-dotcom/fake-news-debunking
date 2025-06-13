from googleapiclient.discovery import build as build_google_api
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os
import re
from dotenv import load_dotenv

# --- Global Clients & Models ---
FACT_CHECK_SERVICE_API = None
IS_FACT_CHECKER_INITIALIZED = False # Tracks overall module init

CLAIM_EXTRACTOR_TOKENIZER = None
CLAIM_EXTRACTOR_MODEL = None
CLAIM_EXTRACTOR_DEVICE = "cpu"
IS_CLAIM_EXTRACTOR_READY = False # Tracks T5 model init

def initialize_fact_checker_resources():
    global FACT_CHECK_SERVICE_API, IS_FACT_CHECKER_INITIALIZED
    global CLAIM_EXTRACTOR_TOKENIZER, CLAIM_EXTRACTOR_MODEL, CLAIM_EXTRACTOR_DEVICE, IS_CLAIM_EXTRACTOR_READY

    if IS_FACT_CHECKER_INITIALIZED and IS_CLAIM_EXTRACTOR_READY: # Check both
        return True

    print("--- Initializing Fact Checker Resources ---")
    fact_check_api_ok = False

    # Initialize Fact Check API Client
    if not FACT_CHECK_SERVICE_API: # Initialize only if not already done
        api_key = os.environ.get("GOOGLE_API_KEY") 
        if not api_key:
            print("WARNING: GOOGLE_API_KEY for Fact Check not set. Fact-checking API will be skipped.")
        else:
            try:
                FACT_CHECK_SERVICE_API = build_google_api("factchecktools", "v1alpha1", developerKey=api_key)
                print("Fact Check API client initialized.")
                fact_check_api_ok = True
            except Exception as e:
                print(f"ERROR initializing Fact Check API: {e}")

    # Initialize T5 Claim Extractor Model
    if not CLAIM_EXTRACTOR_MODEL: 
        try:
            model_name = "Babelscape/t5-base-summarization-claim-extractor"
            print(f"Loading T5 Claim Extractor model: {model_name}...")
            CLAIM_EXTRACTOR_TOKENIZER = T5Tokenizer.from_pretrained(model_name)
            CLAIM_EXTRACTOR_MODEL = T5ForConditionalGeneration.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                CLAIM_EXTRACTOR_DEVICE = "cuda"
                CLAIM_EXTRACTOR_MODEL.to(CLAIM_EXTRACTOR_DEVICE)
                print(f"T5 Claim Extractor moved to {CLAIM_EXTRACTOR_DEVICE}.")
            else:
                print("T5 Claim Extractor will run on CPU (slower).")
            CLAIM_EXTRACTOR_MODEL.eval()
            print("T5 Claim Extractor model initialized successfully.")
            IS_CLAIM_EXTRACTOR_READY = True # Mark T5 as ready
        except Exception as e:
            print(f"ERROR initializing T5 Claim Extractor model: {e}")
    else: 
        IS_CLAIM_EXTRACTOR_READY = True


    # Overall module initialization status
    IS_FACT_CHECKER_INITIALIZED = fact_check_api_ok 
    return IS_FACT_CHECKER_INITIALIZED

def extract_key_claims_with_t5(article_text, article_title=None, max_claims_to_return=2,
                               max_input_length=1024, # Max tokens for T5 input
                               max_output_length=150, # Max tokens for generated claims string
                               min_output_length=20): # Min tokens for generated claims string
    """
    Extracts claims from text using the Babelscape T5 model.
    The model outputs claims as a single string; this function splits it into sentences.
    """
    if not IS_CLAIM_EXTRACTOR_READY or not CLAIM_EXTRACTOR_MODEL or not CLAIM_EXTRACTOR_TOKENIZER:
        print("  T5 Claim Extractor model not initialized. Using heuristic fallback for claim extraction.")
        return extract_key_claims_heuristic_fallback(article_title, article_text, max_claims=max_claims_to_return)

    if not article_text or not article_text.strip():
        return []

    print("  Extracting key claims using T5 model (Babelscape/t5-base-summarization-claim-extractor)...")
    
    # If results are poor, experiment with prefixes like "extract claims: " or "summarize and extract claims: "
    input_text_for_t5 = article_text
    if article_title:
        input_text_for_t5 = f"{article_title.strip()}. {article_text.strip()}"


    try:
        # Tokenize the input text
        tok_input = CLAIM_EXTRACTOR_TOKENIZER.batch_encode_plus(
            [input_text_for_t5], # Must be a list of strings
            max_length=max_input_length,
            return_tensors="pt",
            padding="longest", # Pad to the longest sequence in the batch (or max_length)
            truncation=True
        ).to(CLAIM_EXTRACTOR_DEVICE)

        # Generate claims using the model
        generated_ids = CLAIM_EXTRACTOR_MODEL.generate(
            **tok_input, # Pass tokenized input (input_ids, attention_mask)
            num_beams=4, # From typical T5 summarization settings
            max_length=max_output_length,
            min_length=min_output_length,
            early_stopping=True
        )
        
        # Decode the generated tokens into a string
        claims_string = CLAIM_EXTRACTOR_TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # A regex that handles common sentence terminators.
        potential_claims = re.split(r'(?<=[.!?])\s+(?=[A-Z"“])', claims_string.strip()) # Split on space after terminator if next word starts with Cap/Quote
        if not potential_claims or (len(potential_claims) == 1 and potential_claims[0] == claims_string.strip()):
            # Fallback if the above split doesn't work well (e.g. no standard punctuation)
            potential_claims = re.split(r'\n', claims_string.strip()) 

        extracted_claims = []
        for claim in potential_claims:
            claim_stripped = claim.strip()
            # Filter for reasonably sized, assertive-looking sentences
            # Adjust length constraints as needed based on typical output
            if claim_stripped and 5 < len(claim_stripped.split()) < 60 and \
               not claim_stripped.lower().startswith(("this summary", "in summary", "the claims are", "claims include")):
                extracted_claims.append(claim_stripped)
        
        if not extracted_claims and claims_string:
            if 5 < len(claims_string.split()) < 100: 
                 extracted_claims.append(claims_string)


        final_claims = list(dict.fromkeys(extracted_claims)) 
        
        if final_claims:
            print(f"    T5 Extracted Claims: {final_claims[:max_claims_to_return]}")
        else:
            print(f"    T5 model generated output, but no distinct claims were parsed: '{claims_string[:100]}...'")
        return final_claims[:max_claims_to_return]

    except Exception as e:
        print(f"    Error during T5 claim extraction: {e}")
        return extract_key_claims_heuristic_fallback(article_title, article_text, max_claims=max_claims_to_return)

# Using regular expressions for claim extraction if Transformer model fails.
def extract_key_claims_heuristic_fallback(article_title, article_text, article_description=None, max_claims=2):
    print("    Using heuristic fallback for claim extraction.")
    if not article_text and not article_title and not article_description: return []
    claims = []; 
    if article_text:
        quoted_claims = re.findall(r'["“]([^"”]{25,250})["”]', article_text)
        for qc in quoted_claims:
            if len(qc.split()) > 4 and not qc.lower().startswith(("he said", "she said", "they said")): claims.append(qc)
    if article_description and len(claims) < max_claims:
        desc_sentences = re.split(r'(?<=[.!?])\s+', article_description.strip())
        for sent in desc_sentences:
            if 10 < len(sent) < 250 and len(sent.split()) > 3: claims.append(sent);
            if len(claims) >= max_claims: break
    if article_title and len(claims) < max_claims:
        if len(article_title.split()) > 4 and len(article_title) > 20 and "?" not in article_title:
            if not any(article_title.lower() in c.lower() or c.lower() in article_title.lower() for c in claims): claims.append(article_title)
    if article_text and len(claims) < max_claims:
        sentences = re.split(r'(?<=[.!?])\s+', article_text.strip())
        for sent_idx, sent in enumerate(sentences):
            if len(claims) >= max_claims: break
            if 15 < len(sent) < 250 and len(sent.split()) > 5 and "?" not in sent and sent_idx < 5 :
                if not any(sent.lower() in c.lower() or c.lower() in sent.lower() for c in claims): claims.append(sent)
    final_claims = []; seen_claims_lower = set()
    for claim in claims:
        claim_stripped = claim.strip()
        if claim_stripped and len(claim_stripped.split()) > 3 and claim_stripped.lower() not in seen_claims_lower:
            final_claims.append(claim_stripped); seen_claims_lower.add(claim_stripped.lower())
            if len(final_claims) >= max_claims: break
    return final_claims


def run_fact_check_on_extracted_claims(claims_list):
    if not IS_FACT_CHECKER_INITIALIZED or not FACT_CHECK_SERVICE_API: 
        print("Fact Check service not initialized. Skipping fact checks.")
        return {claim: [{"error": "Fact Check service not initialized"}] for claim in claims_list if claim}
    all_fact_check_results = {}
    if not claims_list: return all_fact_check_results
    print(f"  Attempting to fact-check {len(claims_list)} extracted claim(s)...")
    for claim_text in claims_list:
        if not claim_text: continue
        try:
            request = FACT_CHECK_SERVICE_API.claims().search(query=claim_text, languageCode="en")
            response = request.execute()
            found_claim_reviews = response.get("claims", [])
            formatted_results = []
            if found_claim_reviews:
                for item in found_claim_reviews:
                    if item.get('claimReview'):
                        for review in item['claimReview']:
                            formatted_results.append({
                                "original_query_claim": claim_text,
                                "retrieved_claim_text": item.get('text', 'N/A'),
                                "publisher": review.get('publisher', {}).get('name', 'N/A'),
                                "rating": review.get('textualRating', 'N/A'),
                                "review_url": review.get('url', '#'),
                                "review_date": review.get('reviewDate', 'N/A')})
            all_fact_check_results[claim_text] = formatted_results
        except Exception as e:
            print(f"    Error querying Fact Check API for claim '{claim_text[:50]}...': {e}")
            all_fact_check_results[claim_text] = [{"error": str(e)}]
    return all_fact_check_results


if __name__ == "__main__":
    load_dotenv() # Load .env variables when script is run directly

    if not initialize_fact_checker_resources():
        print("Halting test due to resource initialization errors.")
        exit()

    print("\n--- Testing T5 Claim Extraction & Fact Checking ---")

    sample_article_title_1 = "Scientists Discover Water on Mars Again"
    sample_article_text_1 = """
    Recent rover missions have provided compelling new evidence suggesting the presence of subsurface liquid water on Mars.
    "This is a monumental discovery that could redefine our understanding of potential Martian habitability," stated Dr. Astra Nova, lead scientist.
    The findings, published in the journal 'Cosmic Explorations', detail radar data indicating large aquifers beneath the southern polar ice cap.
    However, some skeptics urge caution, noting that direct sampling is still needed for definitive proof.
    Previous claims about Martian water have faced scrutiny.
    """

    sample_article_title_2 = "New Superfood 'WonderBerry' Cures All Diseases, Experts Claim!"
    sample_article_text_2 = """
    A newly discovered 'WonderBerry' from the remote Amazon is being hailed as a miracle cure for everything from the common cold to cancer.
    "We've never seen anything like it; it's a complete game-changer for medicine," an unnamed researcher was quoted as saying in an online blog.
    The blog post, filled with glowing testimonials, claims that consuming just one berry a day can reverse aging and boost intelligence.
    No peer-reviewed studies have been published yet, but the website selling WonderBerry supplements is experiencing massive demand.
    """

    test_cases = [
        {"title": sample_article_title_1, "text": sample_article_text_1},
        {"title": sample_article_title_2, "text": sample_article_text_2},
        {"title": None, "text": "The earth is flat and the moon landing was faked by Hollywood."}, # Text only
    ]

    for i, case in enumerate(test_cases):
        print(f"\n\n--- Test Case {i+1} ---")
        print(f"Title: {case['title']}")
        print(f"Text (snippet): {case['text'][:150]}...")

        # 1. Extract claims using T5 (or heuristic fallback)
        # We don't have 'article_description' in these test cases, so it will be None
        extracted_claims = extract_key_claims_with_t5(
            case["text"],
            article_title=case["title"],
            max_claims_to_return=2
        )

        if not extracted_claims:
            print("  No claims were extracted for fact-checking.")
            continue

        # 2. Run fact-check on the extracted claims
        fact_check_results = run_fact_check_on_extracted_claims(extracted_claims)

        print("\n  Fact Check API Results:")
        if fact_check_results:
            for original_claim, results_for_claim in fact_check_results.items():
                print(f"    For Your Extracted Claim: \"{original_claim}\"")
                if results_for_claim and not results_for_claim[0].get("error"):
                    for res_item in results_for_claim[:2]: # Show top 2 results per claim
                        print(f"      - Publisher: {res_item.get('publisher', 'N/A')}")
                        print(f"        Rating: {res_item.get('rating', 'N/A')}")
                        print(f"        Retrieved Claim: {res_item.get('retrieved_claim_text', 'N/A')}")
                        print(f"        URL: {res_item.get('review_url', 'N/A')}")
                elif results_for_claim and results_for_claim[0].get("error"):
                    print(f"      - Error: {results_for_claim[0].get('error')}")
                else:
                    print("      - No specific fact-checks found by API for this claim.")
        else:
            print("    No results returned from Fact Check API.")