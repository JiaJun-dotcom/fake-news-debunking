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
from news_extraction import fetch_and_extract_article_data_from_url
from factcheck import *
from sentiment_analysis import initialize_sentiment_analyzers, get_sentiment_vader
import asyncio

load_dotenv()

PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")

# --- Global Client for GenAI --- 
LLAMA_CPP_MODEL = None
GGUF_REPO_ID = "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF"
GGUF_FILENAME = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

# --- Global Initialization Status ---
ALL_RESOURCES_INITIALIZED = False

def initialize_all_module_resources():
    global ALL_RESOURCES_INITIALIZED, LLAMA_CPP_MODEL # Add LLAMA_CPP_MODEL here
    if ALL_RESOURCES_INITIALIZED:
        return True

    print("--- Initializing All Application Resources ---")
    # Initialize each module. They should handle their own client setups.
    # These functions should ideally return True on success, False on failure.
    tactic_ok = initialize_tactic_resources()
    classifier_ok = initialize_classifier_resources()
    fact_checker_ok = initialize_fact_checker_resources()
    sentiment_analyzer_ok = initialize_sentiment_analyzers()
    
    print(f"--- Initializing Llama.cpp Model from HF Repo: {GGUF_REPO_ID}, File: {GGUF_FILENAME} ---")
    llama_cpp_ok = False
    try:
        if torch.cuda.is_available():
            print("CUDA detected by PyTorch. Llama.cpp will attempt GPU offload if compiled with CUDA support.")
        else:
            print("No CUDA GPU detected by PyTorch. Model will run on CPU.")

        LLAMA_CPP_MODEL = Llama.from_pretrained(
            repo_id=GGUF_REPO_ID,
            filename=GGUF_FILENAME,
            n_ctx=4096,  # Context window size. Adjust if needed.
            n_batch=512, # Can help with prompt processing speed
            verbose=False # Set to True for more llama.cpp output
        )
        llama_cpp_ok = True
    except Exception as e:
        print(f"ERROR initializing Llama.cpp model from Hugging Face: {e}")
        print("Ensure the repo_id and filename are correct, and you have internet access.")
        
    if tactic_ok and classifier_ok and fact_checker_ok and sentiment_ok and llama_cpp_ok:
        ALL_RESOURCES_INITIALIZED = True
        print("--- All Application Resources Initialized Successfully ---")
        return True
    else:
        print("--- WARNING: One or more modules failed to initialize. Functionality may be limited. ---")
        ALL_RESOURCES_INITIALIZED = False 
        return False 
        

# --- Generative AI Summarization ---
def generate_final_explanation(analysis_data):
    if not LLAMA_CPP_MODEL:
        error_msg = "Llama.cpp(GGUF) model not initialized. Cannot generate explanation."
        print(f"ERROR: {error_msg}")
        return f"Error: {error_msg}\nRaw Data:\n{json.dumps(analysis_data, indent=2)}"
    
    # --- Construct the User Content for the Chat ---
    # The "system" prompt can be set, and the user prompt will contain the analysis data and task.
    system_message = "You are a helpful news analysis assistant. Your task is to analyze the provided assessment data of a news article and generate a concise, easy-to-understand summary for a general user."
    
    user_prompt_content = "Please analyze the following assessment data:\n\n"
    user_prompt_content += "--- ARTICLE ASSESSMENT DATA ---\n"
    user_prompt_content += f"News Title: {analysis_data.get('processed_title', 'N/A')}\n"
    user_prompt_content += f"Source (if URL provided): {analysis_data.get('source_url', 'N/A')}\n\n"

    # Classifier(fake/real)
    if analysis_data.get('classification'):
        cls_res = analysis_data['classification']
        user_prompt_content += f"1. Our Classifier Prediction: {cls_res.get('prediction', 'N/A').upper()} (Confidence: {cls_res.get('confidence', 0):.0%})\n"
    
    # Sentiment analysis
    if analysis_data.get('overall_article_sentiment'):
        sent_data = analysis_data['overall_article_sentiment']
        sentiment_value_desc = "neutral"

        if sent_data:
            compound = sent_data.get('compound', 0) # compound gives us the total score, giving a gauge of sentiment and magnitude.
            if compound >= 0.05: sentiment_value_desc = f"Positive (VADER score: {compound:.2f})"
            elif compound <= -0.05: sentiment_value_desc = f"Negative (VADER score: {compound:.2f})"
            user_prompt_content += f"2. Overall Sentiment of the text: {sentiment_value_desc}\n"
        else:
            user_prompt_content += "2. Overall Sentiment of the text: Not available.\n"
    
    # Tactic detection
    if analysis_data.get('detected_disinformation_tactics'):
        tactics = analysis_data['detected_disinformation_tactics']
        user_prompt_content += f"3. Detected Disinformation Tactics: {', '.join(tactics) if tactics else 'None detected by current rules'}\n"
    else:
        user_prompt_content += "3. Detected Disinformation Tactics: Not available.\n"
      
    # Vector search for similar articles
    if analysis_data.get('similar_articles_from_db'):
        sim_arts = analysis_data['similar_articles_from_db']
        user_prompt_content += "4. Semantic Similarity to Our Database:\n"
        if sim_arts:
            for art in sim_arts[:2]:
                lbl = "Unknown"
                if art.get('label') == 0: lbl = "Known FAKE"
                elif art.get('label') == 1: lbl = "Known REAL"
                user_prompt_content += f"  - Similar to: \"{art.get('title', 'N/A')}\" ({lbl}, Score: {art.get('similarity_score', 0):.2f})\n"
        else: 
            user_prompt_content += "  - No highly similar articles found.\n"
    
    # Google Fact Check API for claim fact-checking.
    if analysis_data.get('fact_check_api_results'):
        fc_res = analysis_data['fact_check_api_results']
        user_prompt_content += "5. Google Fact Check API Results (for extracted claims):\n"
        fc_empty = True
        if fc_res:
            for claim, results in fc_res.items():
                if results and not results[0].get("error"):
                    fc_empty = False
                    user_prompt_content += f"  For claim: \"{claim[:80]}...\"\n"
                    for res_item in results[:1]: user_prompt_content += f"    - By {res_item.get('publisher', 'N/A')}: {res_item.get('rating', 'N/A')}\n"
        if fc_empty: 
            user_prompt_content += "  - No relevant fact-checks found or claims not suitable.\n"

    user_prompt_content += "\n--- END OF ASSESSMENT DATA ---\n\n"
    user_prompt_content += "Based ONLY on the provided assessment data above, please generate your summary. Follow these instructions for your output:\n"
    user_prompt_content += "1. Start with an overall assessment of the article's likely trustworthiness (e.g., 'appears highly unreliable', 'shows mixed signals', 'appears generally credible but with caveats').\n"
    user_prompt_content += "2. Briefly explain the key reasons for this assessment, referencing specific findings from the data (e.g., classifier prediction, tactics, similarity, fact-checks).\n"
    user_prompt_content += "3. Offer 1-2 actionable pieces of advice or things the user should look out for or consider when reading this article or similar content.\n"
    user_prompt_content += "4. Keep the language clear, objective, and avoid making definitive statements of truth/falsity not directly supported by the provided data (especially the classifier's prediction which is probabilistic)."

    print("\n--- Sending Prompt to Llama.cpp Model (Chat Completion) ---")

    try:
        response = LLAMA_CPP_MODEL.create_chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt_content}
            ],
            max_tokens=700,    
            temperature=0.5,    
            top_p=0.9
        )
        explanation = response["choices"][0]["message"]["content"].strip()
        
        print("--- Llama.cpp Model Response Received ---")
        return explanation
    except Exception as e:
        print(f"Error calling Llama.cpp Model create_chat_completion: {e}")
        return f"Error generating final explanation using Llama.cpp. (Error: {e})"

    
# --- Main Orchestration Function ---
async def analyze_article(user_input_url_or_text):
    if not ALL_RESOURCES_INITIALIZED:
        # Attempt to initialize if not already done 
        if not initialize_all_module_resources():
             return {"error": "Critical resources failed to initialize.",
                    "final_user_explanation": "System error: Could not initialize analysis components."}
    
    print(f"\n\n<<<<< Starting Comprehensive Analysis for Input: {user_input_url_or_text[:100]}... >>>>>")
    article_title = "User Input"
    article_text = None
    article_description = None
    source_url_for_analysis = None # The URL used for tactic source check & prompt
    canonical_url_from_extraction = None # The canonical URL from extraction lib
    
    is_url = bool(re.match(r'^https?://', user_input_url_or_text, re.IGNORECASE))
    if is_url:
        source_url_for_analysis = user_input_url_or_text
        print(f"Input is a URL: {source_url_for_analysis}")
        title, text, description, url_final = await asyncio.to_thread(
            fetch_and_extract_article_data_from_url, user_input_url_or_text
        )
        if text:
            article_title = title if title and title != "Title Not Found" else "Scraped Article"
            article_text = text
            article_description = description
            canonical_url_from_extraction = url_final
        else:
            return {"error": "Could not scrape significant text from URL.",
                    "final_user_explanation": "Could not process the URL to get article content."}
    else:
        print("Input is direct text.")
        article_text = user_input_url_or_text
        first_line = article_text.split('\n', 1)[0]
        if len(first_line) < 120 and len(first_line.split()) < 20: article_title = first_line
        canonical_url_from_extraction = "N/A (Direct Text Input)"

    if not article_text or not article_text.strip():
        return {"error": "No text content to analyze.",
                "final_user_explanation": "No text content provided for analysis."}

    print(f"\n--- Processing ---\nTitle: {article_title[:80]}...\nDescription: {str(article_description)[:80]}...\nText (start): {article_text[:100]}...")

    print("\n[Creating concurrent analysis tasks...]")    
    
    print("\n[Fake/Real Classification]")
    task_classify = asyncio.to_thread(classify_article_text, article_title, article_text)

    print("\n[Semantic Similarity Search]")
    task_vector_search = asyncio.to_thread(find_similar_articles_vector_search, article_title, article_text, 3)
    
    print("\n[Disinformation Tactic Detection]")
    task_tactic_detect = asyncio.to_thread(detect_disinformation_tactics, article_title, article_text, source_url_for_analysis)
    
    print("\n[Overall Sentiment Analysis]")
    task_sentiment = asyncio.to_thread(get_article_sentiment, article_text) # Using "vader" for speed
   
    async def run_fact_checking_pipeline():
        print("  [Starting fact-checking sub-pipeline...]")
        claims = await asyncio.to_thread(extract_key_claims_with_t5, article_text, article_title, article_description, 2)
        if claims:
            print(f"    Claims extracted for fact-checking: {claims}")
            results = await asyncio.to_thread(run_fact_check_on_extracted_claims, claims)
            print("  [Fact-checking sub-pipeline complete.]")
            return results
        else:
            print("    No claims extracted, skipping fact_check API calls.")
            print("  [Fact-checking sub-pipeline complete (no claims).]")
            return {} # Return empty dict if no claims

    task_fact_check_pipeline = run_fact_checking_pipeline()

    print("[Executing analysis tasks concurrently...]")
    results = await asyncio.gather(
        task_classify,
        task_vector_search,
        task_tactic_detect,
        task_sentiment,
        task_fact_check_pipeline,
        return_exceptions=True # Important to catch errors from individual tasks
    )
    print("[All analysis tasks complete.]")
    
    # Unpack results and handle potential errors
    classification_result = results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])}
    similar_articles = results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])}
    detected_tactics = results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])}
    overall_sentiment_result = results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])}
    fact_check_results = results[4] if not isinstance(results[4], Exception) else {"error": str(results[4])}
    
    # --- Compile Data for GenAI ---
    analysis_data_for_genai = {
        "user_input_source": user_input_url_or_text,
        "processed_title": article_title,
        "processed_description": article_description,
        "source_url": canonical_url_from_extraction, # Use canonical URL for prompt
        "classification": classification_result,
        "overall_article_sentiment": overall_sentiment_result,
        "detected_disinformation_tactics": detected_tactics,
        "similar_articles_from_db": similar_articles,
        "fact_check_api_results": fact_check_results,
    }

    print("\n[5. Generating Final Explanation with GenAI]")
    final_explanation = await asyncio.to_thread(generate_final_explanation, analysis_data_for_genai)
    analysis_data_for_genai["final_user_explanation"] = final_explanation
    
    print("\n--- FINAL USER EXPLANATION ---")
    print(final_explanation)
    
    return analysis_data_for_genai


def analyze_article_wrapper(user_input_url_or_text):
    # This runs the async function in a new event loop.
    # Since the caller (like FastAPI's run_in_threadpool) is synchronous, cannot pass in an async function directly.
    return asyncio.run(analyze_article(user_input_url_or_text))

# --- Example Usage for direct script run (testing the async orchestration) ---
if __name__ == "__main__":
    if not initialize_all_module_resources():
        print("Exiting due to resource initialization failure.")
        exit()
    
    test_inputs = [
        """This is a test article. "A shocking new study claims that the sky is actually green," said Dr. Obvious. Many people are talking about it. Some experts agree this is groundbreaking. Check out more at fakerynews.com""",
        "https://www.reuters.com/technology/google-says-generative-ai-search-brings-new-queries-more-usage-2023-05-10/"
    ]

    for user_input in test_inputs:
        # To run the async function from this synchronous __main__ block:
        results = asyncio.run(analyze_article(user_input))
        # Or use the sync wrapper if you prefer:
        # results = master_analyze_article_sync_wrapper(user_input)
        print("\n" + "#" * 70 + "\n")