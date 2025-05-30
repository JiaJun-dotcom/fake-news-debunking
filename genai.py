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
        return True
    except Exception as e:
        print(f"ERROR initializing Llama.cpp model from Hugging Face: {e}")
        print("Ensure the repo_id and filename are correct, and you have internet access.")
        return False

# --- Generative AI Summarization ---
def generate_final_explanation(analysis_data):
    if not LLAMA_CPP_MODEL:
        error_msg = "Llama.cpp(GGUF) model not initialized. Cannot generate explanation."
        print(f"ERROR: {error_msg}")
        # Return the raw data or a simple error message
        return f"Error: {error_msg}\nRaw Data:\n{json.dumps(analysis_data, indent=2)}"
    
    # --- Construct the User Content for the Chat ---
    # The "system" prompt can be set, and the user prompt will contain the analysis data and task.
    system_message = "You are a helpful news analysis assistant. Your task is to analyze the provided assessment data of a news article and generate a concise, easy-to-understand summary for a general user."
    
    user_prompt_content = "Please analyze the following assessment data:\n\n"
    user_prompt_content += "--- ARTICLE ASSESSMENT DATA ---\n"
    user_prompt_content += f"News Title: {analysis_data.get('processed_title', 'N/A')}\n"
    user_prompt_content += f"Source (if URL provided): {analysis_data.get('source_url', 'N/A')}\n\n"

    if analysis_data.get('classification'):
        cls_res = analysis_data['classification']
        user_prompt_content += f"1. Our Classifier Prediction: {cls_res.get('prediction', 'N/A').upper()} (Confidence: {cls_res.get('confidence', 0):.0%})\n"
    # ... (Add ALL other sections: Sentiment, Tactics, Similar Articles, Fact Checks - SAME AS THE PREVIOUS PROMPT)
    # For brevity, I'm not repeating the entire prompt construction here.
    # Ensure you build the full `user_prompt_content` with all data points.
    # Example for one more section:
    if analysis_data.get('detected_disinformation_tactics'):
        tactics = analysis_data['detected_disinformation_tactics']
        user_prompt_content += f"3. Detected Disinformation Tactics: {', '.join(tactics) if tactics else 'None detected by current rules'}\n"
    else:
        user_prompt_content += "3. Detected Disinformation Tactics: Not available.\n"
    # ... include all other data points ...

    user_prompt_content += "\n--- END OF ASSESSMENT DATA ---\n\n"
    user_prompt_content += "Based ONLY on the provided assessment data above, please generate your summary. Follow these instructions for your output:\n"
    user_prompt_content += "1. Start with an overall assessment of the article's likely trustworthiness (e.g., 'appears highly unreliable', 'shows mixed signals', 'appears generally credible but with caveats').\n"
    user_prompt_content += "2. Briefly explain the key reasons for this assessment, referencing specific findings from the data (e.g., classifier prediction, tactics, similarity, fact-checks).\n"
    user_prompt_content += "3. Offer 1-2 actionable pieces of advice or things the user should look out for or consider when reading this article or similar content.\n"
    user_prompt_content += "4. Keep the language clear, objective, and avoid making definitive statements of truth/falsity not directly supported by the provided data (especially the classifier's prediction which is probabilistic)."

    print("\n--- Sending Prompt to Llama.cpp Model (Chat Completion) ---")
    # print(f"User Content (for debugging):\n{user_prompt_content}")

    try:
        response = LLAMA_CPP_MODEL.create_chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt_content}
            ],
            max_tokens=700,     # Max new tokens for the explanation
            temperature=0.3,    # Lower for more focused output
            top_p=0.9,
            # stop=["\n\n", "---"] # Optional stop sequences
        )
        explanation = response["choices"][0]["message"]["content"].strip()
        
        print("--- Llama.cpp Model Response Received ---")
        return explanation
    except Exception as e:
        print(f"Error calling Llama.cpp Model create_chat_completion: {e}")
        return f"Error generating final explanation using Llama.cpp. (Error: {e})"

    
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
    final_explanation = generate_final_explanation_with_llama_cpp(analysis_data_for_genai)

    # Add the GenAI explanation to the full results dictionary
    analysis_data_for_genai["final_user_explanation"] = final_explanation
    
    print("\n--- FINAL USER EXPLANATION ---")
    print(final_explanation)

    return analysis_data_for_genai





