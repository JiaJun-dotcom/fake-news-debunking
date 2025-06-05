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
        result_string = await run_in_threadpool(genai.analyze_article_wrapper, item.content)
        
        if isinstance(result_string, str) and result_string.startswith("Error:"):
            print(f"Analysis wrapper returned an error: {result_string}")
            error_detail = result_string 
            raise HTTPException(status_code=500, detail=error_detail)
        
        if isinstance(result_string, str):
            return result_string 
        else:
            print(f"UNEXPECTED type from analyze_article_wrapper: {type(result_string)}")
            raise HTTPException(status_code=500, detail="Analysis failed: Unexpected internal response format.")

    except Exception as e:
        print(f"UNEXPECTED ERROR in analyze_article_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred.")
        
# --- How to Run This Server ---
# Run from terminal in the directory containing 'main.py':
#    uvicorn main:app --reload --port 8000
#    - 'main': the name of your Python file (main.py).
#    - 'app': the FastAPI instance variable you created (app = FastAPI()).
#    - '--reload': enables auto-reloading when you save changes to the code (for development).

# Access the API:
#    - Send POST requests to: http://127.0.0.1:8000/analyze_article/
#      eg:
#      curl -X POST "http://localhost:8000/analyze_article/" \
#      -H "Content-Type: application/json" \
#      -d '{"content": "This is a sample news article text. Scientists today announced a shocking discovery that cats can indeed fly, but only on Tuesdays. This has been confirmed by multiple unnamed sources and experts agree it is a game changer."}'
# or if want to POST url:
# Replace with: -d '{"content": "https://www.reuters.com/world/europe/shelling-hits-east-ukrainian-city-hours-after-ceasefire-deal-2023-01-06/"}'

# For Video demo, use FastAPI's Automatic Docs (Swagger UI / ReDoc):
# How: Just open http://localhost:8000/docs in your web browser.
# curl -X POST "http://localhost:8000/analyze_article/" \
#       -H "Content-Type: application/json" \
#       -d '{"content": "This is a sample news article text. Scientists today announced a shocking discovery that cats can indeed fly, but only on Tuesdays. This has been confirmed by multiple unnamed sources and experts agree it is a game changer."}'