# FastAPI endpoint to deploy model
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
import time
from pydantic import BaseModel
import genai

ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs once when the FastAPI application starts.
    It loads all necessary resources like ML models.
    """
    print("INFO:     FastAPI application starting up...")
    print("INFO:     Initializing all analysis resources (models, clients, lexicons)...")
    
    try:
        # This function loads all ML models, DB connections, API clients, etc.
        if genai.initialize_all_module_resources():
            ml_resources["is_ready"] = True
            print("INFO:     All analysis resources initialized successfully. API is ready.")
        else:
            raise RuntimeError("CRITICAL: Failed to initialize one or more analysis resources.")
    except Exception as e:
        print(f"CRITICAL: An unexpected error occurred during startup: {e}")
        raise

    yield 

    # This code runs on shutdown.
    print("INFO:     FastAPI application shutting down...")
    ml_resources.clear()
    print("INFO:     Cleared all analysis resources.")


app = FastAPI(
    title="Fake News Debunker API",
    description="Analyzes news articles or text input for potential misinformation, tactics, and provides an AI-generated explanation.",
    version="1.0",
    lifespan=lifespan 
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArticleInput(BaseModel):
    content: str

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"INFO: Request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
    return response

@app.get("/", tags=["Frontend"])
async def serve_frontend():
    """
    Serves the main frontend page.
    Returns a 503 error if the service is still starting up.
    """
    if ml_resources.get("is_ready"):
        return FileResponse("frontend/index.html")
    else:
        raise HTTPException(status_code=503, detail="Service is currently starting up, please try again in a moment.")
    
# --- Analysis Endpoint ---
@app.post("/analyze_article/", tags=["Analysis"])
async def analyze_article_endpoint(item: ArticleInput):
    """
    Endpoint to analyze a news article.
    Accepts a JSON body with a "content" field (URL or text).
    Returns a comprehensive analysis including a GenAI-generated explanation.
    """
    if not ml_resources.get("is_ready"):
        raise HTTPException(status_code=503, detail="Service temporarily unavailable: Critical resources not initialized.")

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
        
app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")

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
