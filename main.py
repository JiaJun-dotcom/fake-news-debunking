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
import threading

ml_resources = {
    "genai_module": None, 
    "is_ready": False,
    "lock": threading.Lock()
}

def initialize_models_if_needed():
    """
    A thread-safe function to initialize resources only once.
    """
    if not ml_resources["is_ready"]:
        with ml_resources["lock"]:
            if not ml_resources["is_ready"]:
                print("INFO:     First request. Lazily importing genai and initializing resources...")
                try:
                    import genai  
                    ml_resources["genai_module"] = genai 

                    if ml_resources["genai_module"].initialize_all_module_resources():
                        ml_resources["is_ready"] = True
                        print("INFO:     All analysis resources initialized successfully. API is ready.")
                    else:
                        print("CRITICAL: Failed to initialize one or more analysis resources.")
                except Exception as e:
                    print(f"CRITICAL: An unexpected error occurred during resource initialization: {e}")
                    ml_resources["is_ready"] = False
                    raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function is super lightweight and ensures a fast startup.
    """
    print("INFO:     FastAPI application starting up...")
    print("INFO:     Server is live. Genai module and models will be loaded on the first analysis request.")
    yield 
    print("INFO:     FastAPI application shutting down...")
    ml_resources.clear()
    print("INFO:     Cleared all analysis resources.")


app = FastAPI(
    title="Fake News Debunker API",
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
    return FileResponse("frontend/index.html")

# Health checker for Render Deployment
@app.get("/health", status_code=200, tags=["System"])
async def health_check():
    """
    Simple endpoint for the hosting platform's health checker.
    """
    return {"status": "ok"}
    
# --- Analysis Endpoint ---
@app.post("/analyze_article/", tags=["Analysis"])
async def analyze_article_endpoint(item: ArticleInput):
    """
    Endpoint to analyze a news article.
    It will trigger module import and model loading on the first call.
    """
    try:
        await run_in_threadpool(initialize_models_if_needed)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service temporarily unavailable: Could not initialize critical resources. Error: {e}")

    if not item.content or not item.content.strip():
        raise HTTPException(status_code=400, detail="Input 'content' cannot be empty.")

    print(f"Received analysis request for content (first 100 chars): {item.content[:100]}...")

    try:
        genai_module = ml_resources["genai_module"]
        result_string = await run_in_threadpool(genai_module.analyze_article_wrapper, item.content)
        
        if isinstance(result_string, str) and result_string.startswith("Error:"):
            print(f"Analysis wrapper returned an error: {result_string}")
            raise HTTPException(status_code=500, detail=result_string)
        
        if isinstance(result_string, str):
            return result_string 
        else:
            print(f"UNEXPECTED type from analyze_article_wrapper: {type(result_string)}")
            raise HTTPException(status_code=500, detail="Analysis failed: Unexpected internal response format.")

    except Exception as e:
        print(f"UNEXPECTED ERROR in analyze_article_endpoint: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred.")
        
app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")

#    uvicorn main:app --reload --port 8000
