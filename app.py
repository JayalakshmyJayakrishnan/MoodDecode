from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import os

app = FastAPI()

# Initialize HuggingFace models
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
crisis_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class TextRequest(BaseModel):
    text: str

@app.post("/analyze_mood")
async def analyze_mood(request: TextRequest):
    result = emotion_classifier(request.text, top_k=1)[0]
    return {"emotion": result['label']}

@app.post("/detect_crisis")
async def detect_crisis(request: TextRequest):
    result = crisis_classifier(request.text, top_k=1)[0]
    crisis_labels = ['sadness', 'fear', 'anger']
    return {"crisis_detected": result['label'] in crisis_labels and result['score'] > 0.85}

@app.post("/summarize")
async def summarize(request: TextRequest):
    summary = summarizer(request.text, max_length=100, min_length=30, do_sample=False)[0]
    return {"summary": summary['summary_text']}
