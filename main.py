from typing import Optional, Dict, Any

import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="ROADSoS Backend")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "ROADSoS backend running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": bool(GEMINI_API_KEY),
    }


class TriageRequest(BaseModel):
    conscious: str
    bleeding: str
    breathing: str


class ChatRequest(BaseModel):
    message: str
    app_context: Optional[str] = None
    location: Optional[Dict[str, Any]] = None


class AssistanceRequest(BaseModel):
    serviceType: str
    vehicleNumber: Optional[str] = None
    vehicleType: Optional[str] = None
    fuelType: Optional[str] = None
    latitude: float
    longitude: float


@app.get("/nearby")
def get_nearby(lat: float, lon: float):
    query = f"""
[out:json][timeout:25];
(
  node["amenity"="hospital"](around:8000,{lat},{lon});
  way["amenity"="hospital"](around:8000,{lat},{lon});
  relation["amenity"="hospital"](around:8000,{lat},{lon});

  node["healthcare"="hospital"](around:8000,{lat},{lon});
  way["healthcare"="hospital"](around:8000,{lat},{lon});
  relation["healthcare"="hospital"](around:8000,{lat},{lon});

  node["amenity"="police"](around:8000,{lat},{lon});
  way["amenity"="police"](around:8000,{lat},{lon});

  node["emergency"="ambulance_station"](around:8000,{lat},{lon});
  way["emergency"="ambulance_station"](around:8000,{lat},{lon});
);
out center;
"""

    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers={"User-Agent": "ROADSoS-Hackathon/1.0"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return {"places": [], "error": str(e)}

    places = []
    for item in data.get("elements", []):
        tags = item.get("tags", {})
        places.append(
            {
                "id": item.get("id"),
                "lat": item.get("lat") or item.get("center", {}).get("lat"),
                "lon": item.get("lon") or item.get("center", {}).get("lon"),
                "name": tags.get("name", "Emergency Service"),
                "type": tags.get(
                    "amenity",
                    tags.get("healthcare", tags.get("emergency", "help")),
                ),
            }
        )

    return {"places": places}


def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file and restart FastAPI.")

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=20,
    )
    result = response.json()

    if "candidates" not in result:
        raise RuntimeError(f"Gemini returned no candidates: {result}")

    return result["candidates"][0]["content"]["parts"][0]["text"].strip()


@app.post("/chat")
def chat(data: ChatRequest):
    location_line = ""
    if data.location:
        lat = data.location.get("latitude")
        lon = data.location.get("longitude")
        if lat and lon:
            location_line = f"\nUser location: https://maps.google.com/?q={lat},{lon}"

    prompt = f"""
You are ROADSoS AI, a concise emergency road-safety assistant for India.

Rules:
- Give practical step-by-step guidance.
- If there is injury, fire, unconsciousness, heavy bleeding, breathing issue, danger, or police risk, tell the user to call 112 immediately.
- Do not diagnose.
- Keep answer short and clear.
- If the message is unclear, ask one direct safety question.

App context: {data.app_context or "ROADSoS emergency app"}
{location_line}

User message:
{data.message}
"""

    try:
        return {"reply": call_gemini(prompt)}
    except Exception as e:
        print("CHAT ERROR:", str(e))
        return {
            "reply": "ROADSoS AI is unavailable right now. If this is serious, call 112 immediately. Move to safety, share your location, and check consciousness, breathing, and bleeding.",
            "error": str(e),
        }


@app.post("/assistance")
def assistance(data: AssistanceRequest):
    print("ASSISTANCE REQUEST:", data.model_dump())

    service_names = {
        "towing": "Tow Truck",
        "mechanic": "Mechanic",
        "fuel": "Fuel Help",
        "charging": "EV Charging Help",
    }
    service_name = service_names.get(data.serviceType, data.serviceType.title())
    maps_link = f"https://maps.google.com/?q={data.latitude},{data.longitude}"

    return {
        "status": "success",
        "message": f"{service_name} request created successfully.",
        "serviceType": data.serviceType,
        "vehicle": {
            "number": data.vehicleNumber or "Not provided",
            "type": data.vehicleType or "Not provided",
            "fuel": data.fuelType or "Not provided",
        },
        "location": {
            "latitude": data.latitude,
            "longitude": data.longitude,
            "maps_link": maps_link,
        },
        "next_step": "Share this request with a nearby provider or emergency contact.",
    }


@app.post("/triage")
def triage(data: TriageRequest):
    prompt = f"""
You are ROADSoS, an emergency road accident triage assistant.

Based ONLY on:
- Conscious: {data.conscious}
- Bleeding: {data.bleeding}
- Breathing: {data.breathing}

Return short guidance.

Format:
Severity: CRITICAL/HIGH/MODERATE/LOW
Action: one short emergency action sentence

Do not give diagnosis.
Keep response under 60 words.
"""

    try:
        ai_text = call_gemini(prompt)
        return {"severity": "AI TRIAGE", "action": ai_text}
    except Exception as e:
        print("TRIAGE ERROR:", str(e))
        return {
            "severity": "FALLBACK",
            "action": "Call 112 immediately if breathing is absent, person is unconscious, or bleeding is heavy.",
            "error": str(e),
        }
