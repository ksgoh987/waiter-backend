# waiter_chat.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os, tempfile
from openai import OpenAI

app = FastAPI()

# ---------- Models ----------
class ChatRequest(BaseModel):
    message: str

# ---------- Health ----------
@app.get("/health")
def health():
    has_key = os.getenv("OPENAI_API_KEY") is not None
    return {"ok": True, "has_openai_key": has_key}

# ---------- Chat ----------
# keep history in memory (simple demo)
chat_history = [
    {"role": "system", "content": "You are WaiterBot, a helpful restaurant waiter. \
Menu: Hainanese Chicken Rice (RM16), Penang Char Kuey Teow (RM18), Iced Lemon Tea (RM6). \
Recommend dishes, answer casually, and remember previous orders in the conversation."}
]

@app.post("/chat")
def chat(req: ChatRequest):
    if client is None:
        return {"reply": f"Echo: {req.message}"}

    try:
        # add user message to history
        chat_history.append({"role": "user", "content": req.message})

        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_history,
        )

        reply = result.choices[0].message.content

        # add assistant reply to history
        chat_history.append({"role": "assistant", "content": reply})

        return {"reply": reply}
    except Exception as e:
        return {"reply": f"(fallback) Echo: {req.message}. Error: {e}"}


# ---------- Transcribe ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if client is None:
        raise HTTPException(status_code=501, detail="OPENAI_API_KEY not set on server")

    try:
        raw = await file.read()
        # write to a temporary file for OpenAI client
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as fh:
            result = client.audio.transcriptions.create(
                model="whisper-1",  # you can also try gpt-4o-mini-transcribe
                file=fh,
            )
        text = getattr(result, "text", "").strip() or "(empty)"
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

