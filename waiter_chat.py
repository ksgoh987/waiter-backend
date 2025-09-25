# waiter_chat.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os, tempfile, uuid
from typing import Dict, List

# ----- Simple in-memory conversation history -----
CHAT_HISTORY: Dict[str, List[dict]] = {}

MENU_TEXT = (
    "Menu:\n"
    "- Hainanese Chicken Rice (RM16)\n"
    "- Penang Char Kuey Teow (RM18)\n"
    "- Iced Lemon Tea (RM6)\n"
)
SYSTEM_PROMPT = (
    "You are WaiterBot, a friendly Malaysian restaurant waiter.\n"
    "REQUIREMENTS:\n"
    "• Always reply in clear, simple ENGLISH only (even if the user speaks another language).\n"
    "• Keep replies short and only suggest items from the menu below.\n"
    "• After your natural reply, ALWAYS include a JSON block on a new line:\n"
    "```json\n"
    "{\"orders\":[{\"name\":\"<menu item>\",\"qty\":<number>}]}\n"
    "```\n"
    "If no order was requested, return {\"orders\":[]}.\n\n"
    f"{MENU_TEXT}"
)





def get_history(session_id: str) -> List[dict]:
    h = CHAT_HISTORY.setdefault(session_id, [])
    if not h:
        h.append({"role": "system", "content": SYSTEM_PROMPT})
    # Cap history size (system + last 18 turns)
    if len(h) > 20:
        CHAT_HISTORY[session_id] = [h[0]] + h[-18:]
        h = CHAT_HISTORY[session_id]
    return h

# ----- FastAPI app -----
app = FastAPI()

# CORS (open for dev; lock down later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# Schemas
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # client should send a stable id; we generate if missing

# ----- Endpoints -----
@app.get("/health")
def health():
    return {"ok": True, "has_openai_key": bool(OPENAI_KEY)}

@app.post("/chat")
def chat(req: ChatRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    session_id = req.session_id or str(uuid.uuid4())
    hist = get_history(session_id)

    try:
        hist.append({"role": "user", "content": req.message})
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=hist,
        )
        reply = result.choices[0].message.content.strip()
        hist.append({"role": "assistant", "content": reply})
        return {"reply": reply, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    try:
        raw = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
            tmp.write(raw)
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
