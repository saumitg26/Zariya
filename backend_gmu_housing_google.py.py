# backend_gmu_housing_gemini.py
import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import google.generativeai as genai
from google.generativeai import models

# ---------------------------
# CONFIGURE GEMINI API
# ---------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY environment variable")
genai.configure(api_key=api_key)

# ---------------------------
# LOAD HOUSING DATA
# ---------------------------
with open("listings.json", "r") as f:  # <-- updated filename
    housing_data = json.load(f)

# ---------------------------
# FASTAPI SETUP
# ---------------------------
app = FastAPI(title="GMU Housing Gemini API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, replace with your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Pydantic models
# ---------------------------
class ChatRequest(BaseModel):
    message: str

class Recommendation(BaseModel):
    title: str
    price: int
    bedrooms: int
    bathrooms: int
    address: str
    reason: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]

# ---------------------------
# Gemini prompt helper
# ---------------------------
def ask_gemini(user_query: str, listings: dict) -> str:
    """
    Send user query + housing data to Gemini, ask for top 3 with explanation,
    and return the raw text response.
    """
    prompt = f"""
You are an expert GMU student housing assistant.

Here is the housing data:
{json.dumps(listings['listings'], indent=2)}

User request: "{user_query}"

Pick the top 3 housing options that best match the user's request.
For each, return a JSON object with fields:
  title, price, bedrooms, bathrooms, address, reason

Explain in the "reason" field why it fits the request.
Return valid JSON only, like:

[
  {{
    "title": "...",
    "price": 0,
    "bedrooms": 0,
    "bathrooms": 0,
    "address": "...",
    "reason": "..."
  }}
]
"""

    response = models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        temperature=0  # deterministic output
    )
    return response.text

# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    raw_response = ask_gemini(request.message, housing_data)

    try:
        recommendations = json.loads(raw_response)
    except Exception as e:
        # If parsing fails, return an empty list and include raw text in reply
        print("Failed to parse Gemini JSON:", e)
        recommendations = []
        return ChatResponse(
            reply=f"Could not parse model output. Raw text:\n{raw_response}",
            recommendations=[]
        )

    reply_text = f"I found {len(recommendations)} housing options matching your request."

    return ChatResponse(reply=reply_text, recommendations=recommendations)
