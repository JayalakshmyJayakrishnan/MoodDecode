# Import necessary libraries from FastAPI, Pydantic, and Hugging Face
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import os

# Initialize the FastAPI application
app = FastAPI()

# Load pre-trained Hugging Face models using transformers' pipeline
# Both emotion and crisis classification use the same emotion model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
crisis_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define the expected request body schema
class TextRequest(BaseModel):
    text: str  # Input text to be processed by endpoints

# Endpoint to analyze emotional tone from input text
@app.post("/analyze_mood")
async def analyze_mood(request: TextRequest):
     # Run emotion classifier and return the top emotion label
    result = emotion_classifier(request.text, top_k=1)[0]
    return {"emotion": result['label']}

# Endpoint to detect potential emotional crisis based on severity and category
@app.post("/detect_crisis")
async def detect_crisis(request: TextRequest):
    # Classify text and check if the predicted emotion indicates a crisis
    result = crisis_classifier(request.text, top_k=1)[0]
    crisis_labels = ['sadness', 'fear', 'anger'] # Emotions considered as potential crisis
    return {"crisis_detected": result['label'] in crisis_labels and result['score'] > 0.85}

# Endpoint to summarize long input text
@app.post("/summarize")
async def summarize(request: TextRequest):
    # Generate a summary with defined constraints
    summary = summarizer(request.text, max_length=100, min_length=30, do_sample=False)[0]
    return {"summary": summary['summary_text']}
