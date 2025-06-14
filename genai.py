# genai.py (Comprehensive News Analyzer)

import os
import json
import time
import re
from dotenv import load_dotenv
from llama_cpp import Llama
import torch
import asyncio

try:
    from tactic_detection import initialize_tactic_resources, \
                                detect_disinformation_tactics
    from search_and_classifier import initialize_classifier_resources, \
                                classify_article_text, find_similar_articles_vector_search, \
                                MONGO_CLIENT as NC_MONGO_CLIENT 
    from news_extraction import fetch_and_extract_article_data_from_url
    from factcheck import initialize_fact_checker_resources, \
                             extract_key_claims_with_t5, run_fact_check_on_extracted_claims
    from sentiment_analysis import initialize_sentiment_analyzers, get_article_sentiment, \
                                   VADER_ANALYZER 
except ImportError as e:
    print(f"ERROR: Could not import from module files. Details: {e}")
    print("Ensure tactic_detector.py, news_classifier.py, news_extraction.py, fact_checker.py, sentiment_analyzer.py exist and are correctly structured.")
    exit()

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---

# --- Global Client for GenAI (Llama.cpp) ---
LLAMA_CPP_MODEL = None
GGUF_REPO_ID = "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF"
GGUF_FILENAME = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

# --- Global Initialization Status ---
ALL_RESOURCES_INITIALIZED = False

def initialize_all_module_resources():
    global ALL_RESOURCES_INITIALIZED, LLAMA_CPP_MODEL
    if ALL_RESOURCES_INITIALIZED:
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
    if not LLAMA_CPP_MODEL:
        error_msg = "Llama.cpp (GGUF) model not initialized. Cannot generate explanation."
        print(f"ERROR: {error_msg}")
        return f"Error: {error_msg}\nRaw Data:\n{json.dumps(analysis_data, indent=2)}"

    # Diagnostic logging
    print("DEBUG: Enter generate_final_explanation with data types:")
    for key, val in analysis_data.items():
        print(f"  - {key}: type={type(val)}, sample={repr(val)[:100]}")

    # Defensive wrapping
    for key in [
        "classification",
        "overall_article_sentiment",
        "detected_disinformation_tactics",
        "similar_articles_summary",
        "fact_check_api_results"
    ]:
        val = analysis_data.get(key)
        if not isinstance(val, dict):
            print(f"DEBUG: Wrapping key '{key}' of type {type(val)} into dict")
            analysis_data[key] = {"value": val}

    system_message = "You are a helpful news analysis assistant..."
    user_prompt_content = "Please analyze the following assessment data for the provided news content:\n\n"
    user_prompt_content += "--- ARTICLE ASSESSMENT DATA ---\n"
    user_prompt_content += f"News Title: {analysis_data.get('processed_title', 'N/A')}\n"
    user_prompt_content += f"Source (if URL provided): {analysis_data.get('source_url', 'N/A')}\n\n"

    # 1. Classifier Prediction
    cls_res = analysis_data["classification"]
    try:
        if cls_res.get("error"):
            user_prompt_content += f"1. Classifier Prediction: Error - {cls_res.get('error')}\n"
        elif "prediction" in cls_res:
            pred = cls_res.get('prediction', 'N/A')
            conf = cls_res.get('confidence', 0)
            user_prompt_content += f"1. Classifier Prediction: **{str(pred).upper()}** (Confidence: {conf:.0%})\n"
        elif "value" in cls_res:
            user_prompt_content += f"1. Classifier Prediction (raw output): {cls_res.get('value')}\n"
        else:
            user_prompt_content += "1. Classifier Prediction: Unexpected format.\n"
    except Exception as e:
        print("ERROR processing classification in generate_final_explanation:", e, cls_res)
        user_prompt_content += "1. Classifier Prediction: Error while formatting result.\n"

    # 2. Overall Sentiment
    sent_data = analysis_data["overall_article_sentiment"]
    try:
        if sent_data.get("error"):
            user_prompt_content += f"2. Overall Sentiment: Error - {sent_data.get('error')}\n"
        elif "compound" in sent_data:
            source = sent_data.get('source', 'Unknown')
            # e.g. if VADER:
            if source.lower() == 'vader':
                compound = sent_data.get('compound', 0)
                if compound >= 0.05:
                    desc = f"Positive (VADER score: {compound:.2f})"
                elif compound <= -0.05:
                    desc = f"Negative (VADER score: {compound:.2f})"
                else:
                    desc = "Neutral"
                user_prompt_content += f"2. Overall Sentiment: {desc}\n"
            else:
                # Other sentiment sources?
                user_prompt_content += f"2. Overall Sentiment: Source={source}, data={sent_data}\n"
        elif "value" in sent_data:
            user_prompt_content += f"2. Overall Sentiment (raw output): {sent_data.get('value')}\n"
        else:
            user_prompt_content += "2. Overall Sentiment: Unexpected format.\n"
    except Exception as e:
        print("ERROR processing sentiment in generate_final_explanation:", e, sent_data)
        user_prompt_content += "2. Overall Sentiment: Error while formatting result.\n"

    # 3. Detected Disinformation Tactics
    tactics = analysis_data["detected_disinformation_tactics"]
    try:
        if tactics.get("error"):
            user_prompt_content += f"3. Detected Disinformation Tactics: Error - {tactics.get('error')}\n"
        elif "tactics" in tactics:
            lst = tactics.get("tactics") or []
            user_prompt_content += f"3. Detected Disinformation Tactics: {', '.join(lst) if lst else 'None detected by current rules'}\n"
        elif "value" in tactics:
            user_prompt_content += f"3. Detected Disinformation Tactics (raw output): {tactics.get('value')}\n"
        else:
            user_prompt_content += "3. Detected Disinformation Tactics: Unexpected format.\n"
    except Exception as e:
        print("ERROR processing tactics in generate_final_explanation:", e, tactics)
        user_prompt_content += "3. Detected Disinformation Tactics: Error while formatting result.\n"

    # 4. Semantic Similarity Summary 
    sim_summary = analysis_data.get("similar_articles_summary", {})
    try:
        user_prompt_content += "4. Semantic Similarity to Our Database:\n"
        if sim_summary.get("error"):
            user_prompt_content += f"  - Error during similarity search: {sim_summary.get('error')}\n"
        else:
            fake_count = sim_summary.get("count_fake", 0)
            real_count = sim_summary.get("count_real", 0)
            if fake_count > 0 and real_count > 0:
                fake_scores_str = ", ".join(sim_summary.get("fake_scores", []))
                real_scores_str = ", ".join(sim_summary.get("real_scores", []))
                user_prompt_content += f"  - Similar to {fake_count} known FAKE (scores: [{fake_scores_str}]) and {real_count} known REAL (scores: [{real_scores_str}]).\n"
            elif fake_count > 0:
                fake_scores_str = ", ".join(sim_summary.get("fake_scores", []))
                user_prompt_content += f"  - Similar to {fake_count} known FAKE (scores: [{fake_scores_str}]).\n"
            elif real_count > 0:
                real_scores_str = ", ".join(sim_summary.get("real_scores", []))
                user_prompt_content += f"  - Similar to {real_count} known REAL (scores: [{real_scores_str}]).\n"
            else:
                if sim_summary.get("raw_count", 0) > 0:
                    user_prompt_content += "  - Similar articles found but no definitive fake/real labels matched.\n"
                else:
                    user_prompt_content += "  - No similar articles found.\n"
    except Exception as e:
        print("ERROR processing semantic similarity in generate_final_explanation:", e, sim_summary)
        user_prompt_content += "4. Semantic Similarity: Error while formatting result.\n"

    # 5. Fact Check Results
    fc_res = analysis_data["fact_check_api_results"]
    try:
        user_prompt_content += "5. Fact Check API Results:\n"
        if fc_res.get("error"):
            user_prompt_content += " - No relevant fact checks available due to an error or unavailable service.\n"
        # 5.2: Structured results present
        elif "results" in fc_res and isinstance(fc_res["results"], dict):
            results_dict = fc_res["results"]
            if not results_dict:
                # Empty dict => no claims or no checks
                user_prompt_content += "  - No relevant fact checks found for the extracted claims.\n"
            else:
                any_valid = False
                # Iterate each claim
                for claim, results_for_claim in results_dict.items():
                    # If not a list or empty list => treat as “no checks” for this claim
                    if not isinstance(results_for_claim, list) or not results_for_claim:
                        user_prompt_content += (
                            f"  For claim: \"{claim[:80]}...\" - No relevant fact checks found.\n"
                        )
                        continue
                    # Filter out error entries
                    valid_items = [
                        item for item in results_for_claim
                        if isinstance(item, dict) and not item.get("error")
                    ]
                    if not valid_items:
                        # All entries reported errors
                        user_prompt_content += (
                            f"  For claim: \"{claim[:80]}...\" - No relevant fact checks found.\n"
                        )
                    else:
                        any_valid = True
                        user_prompt_content += f"  For claim: \"{claim[:80]}...\"\n"
                    # Summarize only the first valid result
                        first = valid_items[0]
                        publisher = first.get("publisher", "N/A")
                        rating = first.get("rating", "N/A")
                        user_prompt_content += (
                            f"    - By {publisher}: Rating: '{rating}'\n"
                        )
                if not any_valid:
                    # None of the claims had valid checks
                    user_prompt_content += "  - No relevant fact checks found for any extracted claim.\n"
        # 5.3: Raw fallback
        elif "value" in fc_res:
            user_prompt_content += f"  - Fact Check raw output: {fc_res.get('value')}\n"
            # 5.4: No recognized keys => no data
        else:
            user_prompt_content += "  - No fact check data available for this article.\n"
    except Exception as e:
        print("ERROR processing fact-check in generate_final_explanation:", e, fc_res)
        user_prompt_content += "5. Fact Check: Error while formatting result.\n"

    user_prompt_content += (
        "Note: If no relevant fact checks were found or the service was unavailable, please explicitly state: "
        "'No relevant or available fact checks yet for this article; please verify independently if needed.'\n\n"
    )
    user_prompt_content += "\n--- END OF ASSESSMENT DATA ---\n\n"
    user_prompt_content += "Task: Based ONLY on the provided assessment data above, generate a concise summary for a general user..."
    user_prompt_content += "\n\nUser-Facing Explanation:"

    # Now call Llama.cpp
    try:
        response = LLAMA_CPP_MODEL.create_chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt_content}
            ],
            max_tokens=2048, temperature=0.25, top_p=0.9,
        )
        explanation = response["choices"][0]["message"]["content"].strip()
        think_delimiter = "</think>"
        final_explanation_text = explanation
        if think_delimiter in explanation:
            parts = explanation.split(think_delimiter, 1)
            if len(parts) > 1:
                # Get the part after the delimiter, then strip leading newlines
                final_explanation_text = parts[1].lstrip('\n ').strip()
            else: # If delimiter is present but split somehow fails to produce 2 parts (edge case)
                final_explanation_text = explanation # Fallback to full text
        
        if final_explanation_text.startswith("User-Facing Explanation:"):
            final_explanation_text = final_explanation_text[len("User-Facing Explanation:"):].lstrip()
        
        return final_explanation_text
    except Exception as e:
        print(f"Error calling Llama.cpp Model create_chat_completion: {e}")
        return f"Error generating final explanation using Llama.cpp. (Details: {e})"



# --- Main Orchestration Function ---
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
    task_sentiment = asyncio.to_thread(get_article_sentiment, article_text)

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

    # 1. Classification normalization
    raw_classify = results[0]
    if isinstance(raw_classify, Exception):
        classification_result = {"error": f"Classifier exception: {raw_classify}"}
    elif isinstance(raw_classify, dict):
        classification_result = raw_classify
    else:
        # wrap other types (string, list, etc.) into a dict
        classification_result = {"value": raw_classify}

    # 2. Similar articles normalization
    raw_similar = results[1]
    if isinstance(raw_similar, Exception):
        similar_articles_raw_output = {"error": f"Vector search exception: {raw_similar}"}
    elif isinstance(raw_similar, dict):
        similar_articles_raw_output = raw_similar
    elif isinstance(raw_similar, list):
        similar_articles_raw_output = {"results": raw_similar}
    else:
        similar_articles_raw_output = {"value": raw_similar}

    # 3. Tactic detection normalization
    raw_tactics = results[2]
    if isinstance(raw_tactics, Exception):
        detected_tactics = {"error": f"TacticDetect exception: {raw_tactics}"}
    elif isinstance(raw_tactics, dict):
        detected_tactics = raw_tactics
    elif isinstance(raw_tactics, list):
        detected_tactics = {"tactics": raw_tactics}
    else:
        detected_tactics = {"value": raw_tactics}

    # 4. Sentiment normalization
    raw_sentiment = results[3]
    if isinstance(raw_sentiment, Exception):
        overall_sentiment_result = {"error": f"Sentiment exception: {raw_sentiment}"}
    elif isinstance(raw_sentiment, dict):
        overall_sentiment_result = raw_sentiment
    else:
        overall_sentiment_result = {"value": raw_sentiment}

    # 5. Fact check normalization
    raw_fact = results[4]
    if isinstance(raw_fact, Exception):
        fact_check_results = {"error": f"FactCheck exception: {raw_fact}"}
    elif isinstance(raw_fact, dict):
        fact_check_results = raw_fact
    elif isinstance(raw_fact, list):
        fact_check_results = {"results": raw_fact}
    else:
        fact_check_results = {"value": raw_fact}


    processed_similar_articles_summary = {
        "count_fake": 0, "fake_scores": [],
        "count_real": 0, "real_scores": [],
        "raw_count": 0 # To know if vector search returned anything at all
    }
    
# Determine actual list of similar articles, if any
    sim_list = None
    if isinstance(similar_articles_raw_output, dict):
        if "error" in similar_articles_raw_output:
            processed_similar_articles_summary["error"] = similar_articles_raw_output["error"]
        elif "results" in similar_articles_raw_output and isinstance(similar_articles_raw_output["results"], list):
            sim_list = similar_articles_raw_output["results"]
        elif "value" in similar_articles_raw_output and isinstance(similar_articles_raw_output["value"], list):
            sim_list = similar_articles_raw_output["value"]
    # else: maybe no list to process
    elif isinstance(similar_articles_raw_output, list):
        # In case you skip normalization, but since you're normalizing, this is unlikely
        sim_list = similar_articles_raw_output

    if sim_list is not None:
        processed_similar_articles_summary["raw_count"] = len(sim_list)
        for doc in sim_list:
            if not isinstance(doc, dict):
                continue
            label = doc.get("label")
            score = doc.get("similarity_score", 0.0)
            if label == 0:
                processed_similar_articles_summary["count_fake"] += 1
                processed_similar_articles_summary["fake_scores"].append(f"{score:.3f}")
            elif label == 1:
                processed_similar_articles_summary["count_real"] += 1
                processed_similar_articles_summary["real_scores"].append(f"{score:.3f}")
    
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
    
    if isinstance(final_explanation, str) and final_explanation.startswith("Error:"):
        analysis_data_for_genai["error"] = "GenAI explanation generation failed."
        
    return analysis_data_for_genai

# Synchronous wrapper for FastAPI
def analyze_article_wrapper(user_input_url_or_text):
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_closed(): 
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError: 
        print("No current event loop, creating a new one.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        # analyze_article is expected to return a dictionary.
        result_dict = loop.run_until_complete(analyze_article(user_input_url_or_text))
    except Exception as e:
        # Catch exceptions from analyze_article or loop.run_until_complete
        print(f"CRITICAL EXCEPTION in analyze_article_wrapper during async call: {e}")
        # import traceback # For debugging
        # print(traceback.format_exc())
        return f"Error: An unexpected critical exception occurred during analysis processing: {str(e)}"

    if isinstance(result_dict, dict):
        if result_dict.get("error"):
            error_message = result_dict.get("final_user_explanation", str(result_dict.get("error")))
            return f"Error: {error_message}"

        final_explanation = result_dict.get("final_user_explanation")
        if isinstance(final_explanation, str):
            if final_explanation.startswith("Error:"): 
                return final_explanation 
            else: 
                return final_explanation
        else:
            return "Error: Analysis completed, but the final explanation is missing or in an unexpected format."
    else:
        return "Error: Core analysis function returned an unexpected data type."


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
        print("\n" + "#" * 70 + "\n")