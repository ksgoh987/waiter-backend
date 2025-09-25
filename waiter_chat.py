from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os, tempfile

# --- Init ---
app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenAI client ---
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# --- Request schema ---
class ChatRequest(BaseModel):
    message: str

# --- Endpoints ---
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(req: ChatRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    try:
        result = client.chat.completions.create(
            model="gpt-4o",   # upgraded model
            messages=[
                {"role": "system", "content": "You are WaiterBot, a friendly restaurant waiter. Suggest menu items and answer questions."},
                {"role": "user", "content": req.message},
            ]
        )
        reply = result.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

@app.post("/transcribe")
async def transcribe(file: bytes = None):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
            tmp.write(file)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as fh:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=fh,
            )
        text = getattr(result, "text", "").strip() or "(empty)"
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
