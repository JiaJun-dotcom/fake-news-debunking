# genai.py (Comprehensive News Analyzer)

import os
import json
import time
import re
from dotenv import load_dotenv
from llama_cpp import Llama
# import vertexai # Not directly used here if Llama.cpp is the GenAI
import torch
import asyncio

try:
    from tactic_detection import initialize_tactic_resources, \
                                detect_disinformation_tactics
    from search_and_classifier import initialize_classifier_resources, \
                                classify_article_text, find_similar_articles_vector_search, \
                                MONGO_CLIENT as NC_MONGO_CLIENT # Expose MONGO_CLIENT for closing
    from news_extraction import fetch_and_extract_article_data_from_url
    from factcheck import initialize_fact_checker_resources, \
                             extract_key_claims_with_t5, run_fact_check_on_extracted_claims
    from sentiment_analysis import initialize_sentiment_analyzers, get_article_sentiment, \
                                   VADER_ANALYZER # Expose VADER if needed by other modules directly
except ImportError as e:
    print(f"ERROR: Could not import from module files. Details: {e}")
    print("Ensure tactic_detector.py, news_classifier.py, news_extraction.py, fact_checker.py, sentiment_analyzer.py exist and are correctly structured.")
    exit()

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
# GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID") # Not needed if only Llama.cpp for GenAI
# GCP_LOCATION = os.environ.get("GCP_LOCATION")   # Not needed if only Llama.cpp for GenAI

# --- Global Client for GenAI (Llama.cpp) ---
LLAMA_CPP_MODEL = None
GGUF_REPO_ID = "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF"
GGUF_FILENAME = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

# --- Global Initialization Status ---
ALL_RESOURCES_INITIALIZED = False

def initialize_all_module_resources():
    global ALL_RESOURCES_INITIALIZED, LLAMA_CPP_MODEL
    if ALL_RESOURCES_INITIALIZED:
        # print("All resources already initialized.") # Can be noisy
        return True

    print("--- Initializing All Application Resources ---")
    tactic_ok = initialize_tactic_resources()
    classifier_ok = initialize_classifier_resources()
    fact_checker_ok = initialize_fact_checker_resources()
    sentiment_analyzer_ok = initialize_sentiment_analyzers()

    print(f"--- Initializing Llama.cpp Model from HF Repo: {GGUF_REPO_ID}, File: {GGUF_FILENAME} ---")
    llama_cpp_ok = False
    try:
        n_gpu_layers_to_offload = -1 if torch.cuda.is_available() else 0
        if n_gpu_layers_to_offload == -1:
            print("CUDA detected by PyTorch. Llama.cpp will attempt to offload all possible layers to GPU.")
        else:
            print("No CUDA GPU detected by PyTorch. Llama.cpp Model will run on CPU.")

        LLAMA_CPP_MODEL = Llama.from_pretrained(
            repo_id=GGUF_REPO_ID,
            filename=GGUF_FILENAME,
            n_ctx=4096,
            n_gpu_layers=n_gpu_layers_to_offload,
            n_batch=512,
            verbose=False
        )
        print(f"Llama.cpp model loaded. GPU layers target: {n_gpu_layers_to_offload}")
        llama_cpp_ok = True
    except Exception as e:
        print(f"ERROR initializing Llama.cpp model from Hugging Face: {e}")
        print(f"Ensure GGUF_MODEL_FILENAME in .env points to a valid GGUF file (e.g., Q4_K_M for 16GB RAM).")

    if tactic_ok and classifier_ok and fact_checker_ok and sentiment_analyzer_ok and llama_cpp_ok:
        ALL_RESOURCES_INITIALIZED = True
        print("--- All Application Resources Initialized Successfully ---")
        return True
    else:
        print("--- WARNING: One or more modules failed to initialize. Functionality may be limited. ---")
        if not tactic_ok: print("  - Tactic detector resources failed.")
        if not classifier_ok: print("  - Classifier/Vector Search resources failed.")
        if not fact_checker_ok: print("  - Fact checker resources failed.")
        if not sentiment_analyzer_ok: print("  - Sentiment analyzer resources failed.")
        if not llama_cpp_ok: print("  - Llama.cpp GenAI model failed.")
        ALL_RESOURCES_INITIALIZED = False
        return False


# --- Generative AI Summarization ---
def generate_final_explanation(analysis_data):
    # (This function remains the same as your last correct version,
    #  it already expects 'similar_articles_summary' in analysis_data)
    if not LLAMA_CPP_MODEL:
        error_msg = "Llama.cpp (GGUF) model not initialized. Cannot generate explanation."
        print(f"ERROR: {error_msg}")
        return f"Error: {error_msg}\nRaw Data (for debugging):\n{json.dumps(analysis_data, indent=2)}"

    system_message = "You are a helpful news analysis assistant..." # Your full system message
    user_prompt_content = "Please analyze the following assessment data for the provided news content:\n\n"
    user_prompt_content += "--- ARTICLE ASSESSMENT DATA ---\n"
    user_prompt_content += f"News Title: {analysis_data.get('processed_title', 'N/A')}\n"
    user_prompt_content += f"Source (if URL provided): {analysis_data.get('source_url', 'N/A')}\n\n"

    # 1. Classifier Prediction
    if analysis_data.get('classification'):
        cls_res = analysis_data['classification']
        if cls_res and not cls_res.get("error"):
            user_prompt_content += f"1. Classifier Prediction: **{cls_res.get('prediction', 'N/A').upper()}** (Confidence: {cls_res.get('confidence', 0):.0%})\n"
        else:
            user_prompt_content += f"1. Classifier Prediction: Error - {cls_res.get('error', 'Not Available')}\n"
    else:
        user_prompt_content += "1. Classifier Prediction: Not Run.\n"

    # 2. Overall Sentiment
    if analysis_data.get('overall_article_sentiment'):
        sent_data = analysis_data['overall_article_sentiment']
        sentiment_value_desc = "neutral"
        if sent_data and not sent_data.get("error"):
            source = sent_data.get('source', 'Unknown')
            if source.lower() == 'vader':
                compound = sent_data.get('compound', 0)
                if compound >= 0.05: sentiment_value_desc = f"Positive (VADER score: {compound:.2f})"
                elif compound <= -0.05: sentiment_value_desc = f"Negative (VADER score: {compound:.2f})"
            user_prompt_content += f"2. Overall Sentiment: {sentiment_value_desc}\n"
        else:
            user_prompt_content += f"2. Overall Sentiment: Error - {sent_data.get('error', 'Not Available')}\n"
    else:
        user_prompt_content += "2. Overall Sentiment: Not Run.\n"

    # 3. Detected Disinformation Tactics
    if analysis_data.get('detected_disinformation_tactics'):
        tactics = analysis_data['detected_disinformation_tactics']
        if isinstance(tactics, list):
            user_prompt_content += f"3. Detected Disinformation Tactics: {', '.join(tactics) if tactics else 'None detected by current rules'}\n"
        elif isinstance(tactics, dict) and tactics.get("error"):
            user_prompt_content += f"3. Detected Disinformation Tactics: Error - {tactics.get('error')}\n"
        else:
             user_prompt_content += "3. Detected Disinformation Tactics: Not available or unexpected format.\n"
    else:
        user_prompt_content += "3. Detected Disinformation Tactics: Not Run.\n"

    # 4. Semantic Similarity (USING THE PREPARED SUMMARY)
    if analysis_data.get('similar_articles_summary'):
        sim_summary = analysis_data['similar_articles_summary']
        user_prompt_content += "4. Semantic Similarity to Our Database:\n"
        if sim_summary.get("error"):
            user_prompt_content += f"  - Error during similarity search: {sim_summary.get('error')}\n"
        else:
            fake_count = sim_summary.get("count_fake", 0)
            real_count = sim_summary.get("count_real", 0)
            if fake_count > 0 and real_count > 0:
                fake_scores_str = ", ".join(sim_summary.get("fake_scores", []))
                real_scores_str = ", ".join(sim_summary.get("real_scores", []))
                user_prompt_content += f"  - This article is semantically similar to **{fake_count} known FAKE article(s)** (scores: [{fake_scores_str}]) and **{real_count} known REAL article(s)** (scores: [{real_scores_str}]) in our database.\n"
            elif fake_count > 0:
                fake_scores_str = ", ".join(sim_summary.get("fake_scores", []))
                user_prompt_content += f"  - This article is semantically similar to **{fake_count} known FAKE article(s)** in our database (scores: [{fake_scores_str}]).\n"
            elif real_count > 0:
                real_scores_str = ", ".join(sim_summary.get("real_scores", []))
                user_prompt_content += f"  - This article is semantically similar to **{real_count} known REAL article(s)** in our database (scores: [{real_scores_str}]).\n"
            else:
                if sim_summary.get("raw_count", 0) > 0:
                     user_prompt_content += "  - Some similar articles were found, but their fake/real status was not definitively matched to known labels in the top results, or no strong similarities were found to labeled articles.\n"
                else:
                     user_prompt_content += "  - No highly similar articles were found in our database.\n"
    else:
        user_prompt_content += "4. Semantic Similarity to Our Database: Analysis not available or no results.\n"

    # 5. Fact Check API Results
    if analysis_data.get('fact_check_api_results'):
        fc_res = analysis_data['fact_check_api_results']
        user_prompt_content += "5. Google Fact Check API Results (for automatically extracted claims):\n"
        fc_empty = True
        if fc_res and isinstance(fc_res, dict):
            for claim, results_for_claim in fc_res.items():
                if results_for_claim and not (isinstance(results_for_claim, list) and results_for_claim[0].get("error")):
                    fc_empty = False
                    user_prompt_content += f"  For claim: \"{claim[:80]}...\"\n"
                    for res_item in results_for_claim[:1]:
                        user_prompt_content += f"    - By {res_item.get('publisher', 'N/A')}: Rating: '{res_item.get('rating', 'N/A')}'\n"
                elif results_for_claim and isinstance(results_for_claim, list) and results_for_claim[0].get("error"):
                    user_prompt_content += f"  For claim: \"{claim[:80]}...\" - Error during fact-check: {results_for_claim[0].get('error')}\n"
                    fc_empty = False
        elif isinstance(fc_res, dict) and fc_res.get("error"):
             user_prompt_content += f"  Error during fact-checking process: {fc_res.get('error')}\n"
             fc_empty = False
        if fc_empty:
            user_prompt_content += "  - No relevant fact-checks found by the API for the extracted claims, or claims were not suitable/extracted.\n"
    else:
        user_prompt_content += "5. Google Fact Check API Results: Not Run or No Claims.\n"

    user_prompt_content += "\n--- END OF ASSESSMENT DATA ---\n\n"
    user_prompt_content += "Task: Based ONLY on the provided assessment data above, generate a concise summary for a general user..." # Your full task instructions
    user_prompt_content += "\n\nUser-Facing Explanation:"

    try:
        response = LLAMA_CPP_MODEL.create_chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt_content}
            ],
            max_tokens=700, temperature=0.25, top_p=0.9,
        )
        explanation = response["choices"][0]["message"]["content"].strip()
        return explanation
    except Exception as e:
        print(f"Error calling Llama.cpp Model create_chat_completion: {e}")
        return f"Error generating final explanation using Llama.cpp. (Details: {e})"


# --- Main Orchestration Function (Async Internals) ---
async def analyze_article(user_input_url_or_text): 
    if not ALL_RESOURCES_INITIALIZED:
         return {"error": "Critical resources not initialized.",
                "final_user_explanation": "System error: Analysis components not ready."}

    print(f"\n\n<<<<< Starting ASYNC Analysis: {user_input_url_or_text[:100]}... >>>>>")
    article_title = "User Input"
    article_text = None
    article_description = None
    source_url_for_analysis = None
    canonical_url_from_extraction = None

    is_url = bool(re.match(r'^https?://', user_input_url_or_text, re.IGNORECASE))
    if is_url:
        source_url_for_analysis = user_input_url_or_text
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
        # print("Input is direct text.") # Can be noisy
        article_text = user_input_url_or_text
        first_line = article_text.split('\n', 1)[0]
        if len(first_line) < 120 and len(first_line.split()) < 20: article_title = first_line
        canonical_url_from_extraction = "N/A (Direct Text Input)"

    if not article_text or not article_text.strip():
        return {"error": "No text content to analyze.",
                "final_user_explanation": "No text content provided for analysis."}

    # --- Create tasks for concurrent execution ---
    task_classify = asyncio.to_thread(classify_article_text, article_title, article_text)
    task_vector_search_raw = asyncio.to_thread(find_similar_articles_vector_search, article_title, article_text, 5)
    task_tactic_detect = asyncio.to_thread(detect_disinformation_tactics, article_title, article_text, source_url_for_analysis)
    task_sentiment = asyncio.to_thread(get_article_sentiment, article_text, "vader")

    async def run_fact_checking_pipeline_async():
        claims = await asyncio.to_thread(extract_key_claims_with_t5, article_text, article_title, article_description, 2)
        return await asyncio.to_thread(run_fact_check_on_extracted_claims, claims if claims else [])
    task_fact_check_pipeline = run_fact_checking_pipeline_async()

    results = await asyncio.gather(
        task_classify,
        task_vector_search_raw,
        task_tactic_detect,
        task_sentiment,
        task_fact_check_pipeline,
        return_exceptions=True
    )

    classification_result = results[0] if not isinstance(results[0], Exception) else {"error": f"Classifier: {str(results[0])}"}
    similar_articles_raw_output = results[1]
    detected_tactics = results[2] if not isinstance(results[2], Exception) else {"error": f"TacticDetect: {str(results[2])}"}
    overall_sentiment_result = results[3] if not isinstance(results[3], Exception) else {"error": f"Sentiment: {str(results[3])}"}
    fact_check_results = results[4] if not isinstance(results[4], Exception) else {"error": f"FactCheck: {str(results[4])}"}

    processed_similar_articles_summary = {
        "count_fake": 0, "fake_scores": [],
        "count_real": 0, "real_scores": [],
        "raw_count": 0 # To know if vector search returned anything at all
    }
    if isinstance(similar_articles_raw_output, list):
        processed_similar_articles_summary["raw_count"] = len(similar_articles_raw_output)
        for doc in similar_articles_raw_output:
            label = doc.get("label")
            score = doc.get("similarity_score", 0.0)
            if label == 0: 
                processed_similar_articles_summary["count_fake"] += 1
                processed_similar_articles_summary["fake_scores"].append(f"{score:.3f}")
            elif label == 1: 
                processed_similar_articles_summary["count_real"] += 1
                processed_similar_articles_summary["real_scores"].append(f"{score:.3f}")
    elif isinstance(similar_articles_raw_output, dict) and "error" in similar_articles_raw_output:
        processed_similar_articles_summary["error"] = similar_articles_raw_output["error"]
    # *** END OF PROCESSING similar_articles_raw_output ***

    analysis_data_for_genai = {
        "user_input_source": user_input_url_or_text,
        "processed_title": article_title,
        "processed_description": article_description,
        "source_url": canonical_url_from_extraction,
        "classification": classification_result,
        "overall_article_sentiment": overall_sentiment_result,
        "detected_disinformation_tactics": detected_tactics,
        "similar_articles_summary": processed_similar_articles_summary, 
        "fact_check_api_results": fact_check_results,
    }

    print("\n[Generating Final Explanation with GenAI]") 
    final_explanation = await asyncio.to_thread(generate_final_explanation, analysis_data_for_genai)
    analysis_data_for_genai["final_user_explanation"] = final_explanation
        
    return analysis_data_for_genai

# Synchronous wrapper for FastAPI
def analyze_article_wrapper(user_input_url_or_text):
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_closed(): 
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError: # No current event loop in this thread
        print("No current event loop, creating a new one.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(analyze_article(user_input_url_or_text))
    # print(f"Sync wrapper finished for: {user_input_url_or_text[:50]}")
    return result


# --- Example Usage for direct script run ---
if __name__ == "__main__":
    
    initialize_all_module_resources()
    
    test_inputs = [
        "https://www.reuters.com/business/finance/jpmorgan-ceo-jamie-dimon-tells-fox-business-us-debt-could-cause-bond-turmoil-2025-06-02/"
    ]

    for user_input in test_inputs:
        results = analyze_article_wrapper(user_input) # Use the sync wrapper for __main__
        print("\n--- Output for input:", user_input[:60], "...")
        print(json.dumps(results.get("final_user_explanation", "No explanation."), indent=2))
        # To see full data: print(json.dumps(results, indent=2))
        print("\n" + "#" * 70 + "\n")