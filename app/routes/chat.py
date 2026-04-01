from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models import ChatRequest, Message
from app.services.claude_client import chat_stream

router = APIRouter()

# In-memory session store: {session_id: [{"role": ..., "content": ...}]}
sessions: dict[str, list[dict]] = {}


@router.post("/chat")
def chat(req: ChatRequest):
    if req.session_id not in sessions:
        sessions[req.session_id] = []

    history = sessions[req.session_id]
    history.append({"role": "user", "content": req.message})

    # Dùng để gom toàn bộ assistant response sau khi stream xong
    collected: list[str] = []

    def generate():
        for chunk in chat_stream(history):
            collected.append(chunk)
            yield chunk
        # Lưu assistant response vào history sau khi stream xong
        full_response = "".join(collected)
        history.append({"role": "assistant", "content": full_response})

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@router.get("/history/{session_id}", response_model=list[Message])
def get_history(session_id: str):
    history = sessions.get(session_id, [])
    return [Message(role=m["role"], content=m["content"]) for m in history]


@router.delete("/history/{session_id}")
def delete_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session không tồn tại")
    del sessions[session_id]
    return {"detail": "Đã xóa session"}
