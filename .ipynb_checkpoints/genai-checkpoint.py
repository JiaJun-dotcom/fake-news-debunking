import os
import json
import time
from dotenv import load_dotenv
from llama_cpp import Llama
import vertexai
import torch
from tactic_detection import *
from ./data_preprocessing/vector_embeddings import *
from search_and_classifier import *

load_dotenv()
from llama_cpp import Llama

PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")

# --- Global Client for GenAI --- 
LLAMA_CPP_MODEL = None
GGUF_REPO_ID = "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF"
GGUF_FILENAME = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

def initialize_llama_cpp_resources():
    global LLAMA_CPP_MODEL
    print(f"--- Initializing Llama.cpp Model from HF Repo: {GGUF_REPO_ID}, File: {GGUF_FILENAME} ---")

    try:
        n_gpu_layers_to_offload = 0
        if torch.cuda.is_available():
            print("CUDA detected by PyTorch. Llama.cpp will attempt GPU offload if compiled with CUDA support.")
            # For BF16 on a 7B/8B model, you'll need significant VRAM to offload many layers.
            # Start with a smaller number or 0 if unsure about VRAM.
            # If using a Q4_K_M (4-bit) GGUF, you can offload more.
            n_gpu_layers_to_offload = -1 # Try to offload all possible, llama.cpp will manage.
                                        # Or a specific number like 20, 32 etc.
                                        # If BF16 is too large, this might still fail on GPU.
                                        # For 16GB system RAM, BF16 will primarily run on CPU.
        else:
            print("No CUDA GPU detected by PyTorch. Model will run on CPU.")


        LLAMA_CPP_MODEL = Llama.from_pretrained(
            repo_id=GGUF_REPO_ID,
            filename=GGUF_FILENAME,
            n_ctx=4096,  # Context window size. Adjust if needed.
            n_gpu_layers=n_gpu_layers_to_offload,
            # n_batch=512, # Can sometimes help with prompt processing speed
            verbose=False # Set to True for more llama.cpp output
        )
        print(f"Llama.cpp model loaded successfully. GPU layers target: {n_gpu_layers_to_offload}")
        return True
    except Exception as e:
        print(f"ERROR initializing Llama.cpp model from Hugging Face: {e}")
        print("Ensure the repo_id and filename are correct, and you have internet access.")
        print("If it's a memory issue with BF16, consider a quantized GGUF (e.g., Q4_K_M).")
        return False

# --- Generative AI Summarization ---
def generate_final_explanation(analysis_data):
    if not GENERATIVE_MODEL:
        error_msg = "Generative model not initialized. Cannot generate explanation."
        print(f"ERROR: {error_msg}")
        # Return the raw data or a simple error message
        return f"Error: {error_msg}\nRaw Data:\n{json.dumps(analysis_data, indent=2)}"
    
    # Construct the prompt
    prompt = "You are a helpful news analysis assistant. Analyze the following assessment of a news article and provide a concise, easy-to-understand summary for the user. Highlight key reasons for concern or confidence regarding the article's trustworthiness. Explain what specific things the user should look out for based on the findings. Be objective and stick to the provided data.\n\n"
    prompt += "--- ARTICLE ASSESSMENT DATA ---\n"
    prompt += f"News Title: {analysis_data.get('processed_title', 'N/A')}\n"
    prompt += f"Source (if URL provided): {analysis_data.get('source_url', 'N/A')}\n\n"


    if analysis_data.get('classification'):
        cls_res = analysis_data['classification']
        prompt += f"1. Our Classifier Prediction: {cls_res.get('prediction', 'N/A').upper()} (Confidence: {cls_res.get('confidence', 0):.0%})\n"
    else:
        prompt += "1. Our Classifier Prediction: Not available.\n"

    # Using VADER sentiment if available from tactic detector's output
    if analysis_data.get('overall_vader_sentiment'):
        vader_s = analysis_data['overall_vader_sentiment']
        sentiment_desc = "neutral"
        if vader_s['compound'] >= 0.05: sentiment_desc = f"positive (VADER score: {vader_s['compound']:.2f})"
        elif vader_s['compound'] <= -0.05: sentiment_desc = f"negative (VADER score: {vader_s['compound']:.2f})"
        prompt += f"2. Overall Sentiment of the text: {sentiment_desc}\n"
    else:
        prompt += "2. Overall Sentiment of the text: Not available.\n"


    if analysis_data.get('detected_disinformation_tactics'):
        tactics = analysis_data['detected_disinformation_tactics']
        prompt += f"3. Detected Disinformation Tactics: {', '.join(tactics) if tactics else 'None detected by current rules'}\n"
    else:
        prompt += "3. Detected Disinformation Tactics: Not available.\n"


    if analysis_data.get('similar_articles_from_db'):
        prompt += "4. Semantic Similarity to Articles in Our Database:\n"
        similar_arts = analysis_data['similar_articles_from_db']
        if similar_arts:
            for sim_art in similar_arts[:2]: # Show top 2 for brevity
                label = "Unknown Label"
                # Assuming label is 0 for fake, 1 for real in your DB
                if sim_art.get('label') == 0: label = "Known FAKE"
                elif sim_art.get('label') == 1: label = "Known REAL"
                prompt += f"  - Similar to: \"{sim_art.get('title', 'N/A')}\" ({label}, Similarity: {sim_art.get('similarity_score', 0):.2f})\n"
        else:
            prompt += "  - No highly similar articles found in our database.\n"
    else:
        prompt += "4. Semantic Similarity to Articles in Our Database: Not available.\n"


    if analysis_data.get('fact_check_api_results'):
        prompt += "5. Google Fact Check API Results (for automatically extracted claims from the article):\n"
        fc_results = analysis_data['fact_check_api_results']
        fc_empty = True
        if fc_results:
            for claim_text, results_for_claim in fc_results.items():
                if results_for_claim and not results_for_claim[0].get("error"): # Check if results exist and no error
                    fc_empty = False
                    prompt += f"  For claim: \"{claim_text[:100]}...\"\n"
                    for res in results_for_claim[:1]: # Show top 1 fact check per claim
                        prompt += f"    - Publisher: {res.get('publisher', 'N/A')}, Rating: {res.get('rating', 'N/A')}\n" # Removed URL for brevity in prompt
        if fc_empty:
            prompt += "  - No relevant fact-checks found by the API for the extracted claims, or claims were not suitable for checking.\n"
    else:
        prompt += "5. Google Fact Check API Results: Not available or no claims extracted.\n"

    prompt += "\n--- END OF ASSESSMENT DATA ---\n\n"
    prompt += "Task: Based ONLY on the provided assessment data above, generate a concise summary for a general user. \n"
    prompt += "1. Start with an overall assessment of the article's likely trustworthiness (e.g., 'appears highly unreliable', 'shows mixed signals', 'appears generally credible but with caveats').\n"
    prompt += "2. Briefly explain the key reasons for this assessment, referencing specific findings from the data (e.g., classifier prediction, tactics, similarity, fact-checks).\n"
    prompt += "3. Offer 1-2 actionable pieces of advice or things the user should look out for or consider when reading this article or similar content.\n"
    prompt += "4. Keep the language clear, objective, and avoid making definitive statements of truth/falsity not directly supported by the provided data (especially the classifier's prediction which is probabilistic).\n"
    prompt += "Output:"


    print("\n--- Sending Prompt to Generative Model ---")
    
    
# --- Main Orchestration Function ---
def analyze_article(user_input_url_or_text):
    print(f"\n\n<<<<< Starting Comprehensive Analysis for Input: {user_input_url_or_text[:100]}... >>>>>")
    article_title = "User Input"
    article_text = None
    source_url_for_tactics_and_prompt = None 

    if user_input_url_or_text.startswith("http://") or user_input_url_or_text.startswith("https://"):
        source_url_for_tactics_and_prompt = user_input_url_or_text
        print(f"Input is a URL: {source_url_for_tactics_and_prompt}")
        scraped_title, scraped_text = scrape_text_from_url(user_input_url_or_text) # From news_classifier 
        if scraped_text:
            article_title = scraped_title if scraped_title and scraped_title != "Title Not Found" else "Scraped Article"
            article_text = scraped_text
        else:
            return {"error": "Could not scrape content from URL.", "final_user_explanation": "Could not process the URL."}
    else:
        print("Input is text content.")
        article_text = user_input_url_or_text
        # Try to infer title
        first_line = article_text.split('\n', 1)[0]
        if len(first_line) < 120 and len(first_line.split()) < 20: article_title = first_line

    if not article_text or not article_text.strip():
        return {"error": "No text content to analyze.", "final_user_explanation": "No text content provided for analysis."}

    print(f"\n--- Processing ---\nTitle: {article_title[:100]}...\nText (start): {article_text[:200]}...")

    # --- Run All Analysis Processes ---
    # These functions are imported from your other .py files
    print("\n[Fake/Real Classification]")
    classification_result = classify_article_text(article_title, article_text)

    print("\n[Semantic Similarity Search]")
    similar_articles = find_similar_articles_vector_search(article_title, article_text, num_results=3)

    print("\n[Disinformation Tactic Detection]")
    detected_tactics = detect_disinformation_tactics(article_title, article_text, source_url=source_url_for_tactics_and_prompt)
    
    # Get overall VADER sentiment
    overall_vader_sentiment = None
    if VADER_ANALYZER and article_text: # VADER_ANALYZER is initialized in tactic_detector's init
         overall_vader_sentiment = VADER_ANALYZER.polarity_scores(article_text)


    print("\n[Fact Checking Extracted Claims]")
    claims_to_fact_check = extract_key_claims_for_fact_check(article_text, max_claims=2)
    fact_check_results = run_fact_check_on_claims(claims_to_fact_check)


    # --- Compile Data for GenAI ---
    analysis_data_for_genai = {
        "user_input_source": user_input_url_or_text,
        "processed_title": article_title,
        "source_url": source_url_for_tactics_and_prompt, 
        "classification": classification_result,
        "overall_vader_sentiment": overall_vader_sentiment,
        "detected_disinformation_tactics": detected_tactics,
        "similar_articles_from_db": similar_articles,
        "fact_check_api_results": fact_check_results,
    }

    # --- 5. Generate Final Explanation ---
    print("\n[Generating Final Explanation with GenAI]")
    final_explanation = generate_final_explanation_with_genai(analysis_data_for_genai)

    # Add the GenAI explanation to the full results dictionary
    analysis_data_for_genai["final_user_explanation"] = final_explanation
    
    print("\n--- FINAL USER EXPLANATION ---")
    print(final_explanation)

    return analysis_data_for_genai





