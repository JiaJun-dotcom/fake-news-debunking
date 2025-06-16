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

ml_resources = {
    "is_ready": False,
    "lock": threading.Lock()
}

def initialize_models_if_needed():
    """
    A thread-safe function to initialize resources only once.
    This is the core of the lazy-loading pattern.
    """
    # Use a non-blocking check first for performance.
    if not ml_resources["is_ready"]:
        with ml_resources["lock"]:
            if not ml_resources["is_ready"]:
                print("INFO:     First request received. Initializing all analysis resources now...")
                try:
                    if genai.initialize_all_module_resources():
                        ml_resources["is_ready"] = True
                        print("INFO:     All analysis resources initialized successfully. API is ready.")
                    else:
                        print("CRITICAL: Failed to initialize one or more analysis resources.")
                        # Keep is_ready as False so future requests might try again
                except Exception as e:
                    print(f"CRITICAL: An unexpected error occurred during resource initialization: {e}")
                    ml_resources["is_ready"] = False
                    raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function now does very little at startup, ensuring the server starts quickly.
    """
    print("INFO:     FastAPI application starting up...")
    ml_resources["is_ready"] = False 
    print("INFO:     Server is live. Models will be loaded on the first analysis request.")
    
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
    Serves the main frontend page. This will now always work instantly.
    """
    return FileResponse("frontend/index.html")
    
# --- Analysis Endpoint ---
@app.post("/analyze_article/", tags=["Analysis"])
async def analyze_article_endpoint(item: ArticleInput):
    """
    Endpoint to analyze a news article.
    It will trigger model loading on the first call.
    """
    try:
        # This function will block and load models only if they haven't been loaded yet.
        # It's run in a threadpool to avoid blocking the main FastAPI event loop.
        await run_in_threadpool(initialize_models_if_needed)
    except Exception as e:
        # If initialization failed, return a 503 error.
        raise HTTPException(status_code=503, detail=f"Service temporarily unavailable: Could not initialize critical resources. Error: {e}")

    # After the check, we know the models are ready (or an exception was raised).
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
        # Check if the exception is one we've already handled
        if isinstance(e, HTTPException):
            raise e
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
