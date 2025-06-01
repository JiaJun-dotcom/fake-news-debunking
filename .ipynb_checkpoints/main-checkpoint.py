# FastAPI endpoint to deploy model
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool # To run blocking IO/CPU tasks
import os
import time
from pydantic import BaseModel
import genai

app = FastAPI(
title="Fake News Debunker API",
    description="Analyzes news articles or text input for potential misinformation, tactics, and provides an AI-generated explanation.",
    version="1.0"
)

class ArticleInput(BaseModel):
    content: str # Can be URL or text 
    
RESOURCES_INITIALIZED_SUCCESSFULLY = False

@app.on_event("startup")
async def startup_event():
    """
    This function is executed once when the FastAPI application starts.
    Loads models, initializes database connections,
    and sets up any other resources needed by application.
    """
    global RESOURCES_INITIALIZED_SUCCESSFULLY
    print("FastAPI application starting up...")
    print("Initializing all analysis resources (models, clients, lexicons)...")
    # This function should load all ML models, DB connections, API clients, etc.
    if genai.initialize_all_module_resources():
        RESOURCES_INITIALIZED_SUCCESSFULLY = True
        print("All analysis resources initialized successfully. API is ready.")
    else:
        RESOURCES_INITIALIZED_SUCCESSFULLY = False
        print("CRITICAL ERROR: Failed to initialize one or more analysis resources during startup.")
        
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    This middleware intercepts every HTTP request.
    It logs basic information about the request and its processing time.
    """
    start_time = time.time()
    response = await call_next(request) # Process the request
    process_time = time.time() - start_time
    # Log request details (can be expanded)
    print(f"INFO: Request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
    return response

@app.get("/", tags=["General"])
async def read_root():
    """
    A simple GET endpoint at the root URL.
    Can be used as a basic health check or to provide API information.
    """
    if RESOURCES_INITIALIZED_SUCCESSFULLY:
        return {"message": "Welcome to the Fake News Debunker API. Resources are initialized."}
    else:
        return {"message": "Welcome to the Fake News Debunker API. WARNING: Resources failed to initialize properly."}

# --- Analysis Endpoint ---
@app.post("/analyze_article/", tags=["Analysis"])
async def analyze_article_endpoint(item: ArticleInput):
    """
    Endpoint to analyze a news article.
    Accepts a JSON body with a "content" field (URL or text).
    Returns a comprehensive analysis including a GenAI-generated explanation.
    """
    # Check if resources were initialized successfully during startup.
    if not RESOURCES_INITIALIZED_SUCCESSFULLY:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable: Critical resources not initialized.")

    # Validate input content from the Pydantic model.
    if not item.content or not item.content.strip():
        raise HTTPException(status_code=400, detail="Input 'content' cannot be empty.")

    print(f"Received analysis request for content (first 100 chars): {item.content[:100]}...")

    try:
        analysis_results = await run_in_threadpool(genai.analyze_article_wrapper, item.content)
        # Ensures it can process multiple requests concurrently using FastAPI's run_in_threadpool function
        # But it expects blocking synchronous code, so have to wrap the async processing of the functions(components)
        # in a synchronous callable argument.
        
        if analysis_results.get("error"):
            print(f"Analysis function returned an error: {analysis_results.get('error')}")
            raise HTTPException(status_code=500, detail=analysis_results.get("final_user_explanation", "Analysis failed internally."))
            
        return {
            "final_user_explanation": analysis_results.get("final_user_explanation", "No explanation generated."),
            "full_analysis_data": analysis_results 
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"UNEXPECTED ERROR during analysis: {e}") 
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
        
# --- How to Run This Server ---
# 1. Save this file as 'main.py'.
# 2. Make sure 'genai.py' (your comprehensive script) and all its dependencies
#    (lexicons.json, model directories) are in the correct places.
# 3. Ensure all Python packages are installed (fastapi, uvicorn, requests, transformers, etc.).
# 4. Run from your terminal in the directory containing 'main.py':
#    uvicorn main:app --reload
#    - 'main': the name of your Python file (main.py).
#    - 'app': the FastAPI instance variable you created (app = FastAPI()).
#    - '--reload': enables auto-reloading when you save changes to the code (for development).
# 5. Access the API:
#    - Documentation: Open your browser to http://127.0.0.1:8000/docs (Swagger UI)
#                     or http://127.0.0.1:8000/redoc (ReDoc)
#    - Send POST requests to: http://127.0.0.1:8000/analyze_article/
#      with a JSON body like: {"content": "Your news article text or URL here"}