# FastAPI endpoint to deploy model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import genai

app = FastAPI()

class ArticleInput(BaseModel):
    content: str # Can be URL or text 
    
