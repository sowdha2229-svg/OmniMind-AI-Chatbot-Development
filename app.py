import json
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

import config
from database import (
    init_db,
    get_conversations,
    get_conversation,
    create_conversation,
    update_conversation_title,
    delete_conversation,
    clear_conversation_messages,
    add_message,
    delete_last_assistant_message
)
from gemini_service import gemini_service
from conversation import (
    QUICK_ACTION_CARDS,
    auto_generate_title,
    prepare_conversation_context
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("omnimind.app")

# Initialize database schema
init_db()

app = FastAPI(
    title="OmniMind AI API",
    description="High-performance backend for OmniMind AI – Universal Intelligent Companion",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Models
class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str = Field(..., min_length=1, description="User prompt text")
    style: Optional[str] = Field("default", description="Explanation style")
    language: Optional[str] = Field("English", description="Target language")
    custom_instructions: Optional[str] = None
    model: Optional[str] = None
    category: Optional[str] = "general"

class TitleUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)

class ApiKeyUpdateRequest(BaseModel):
    api_key: str = Field(..., min_length=10)

class FormalizeRequest(BaseModel):
    text: str = Field(..., min_length=1)

@app.post("/api/formalize")
async def formalize_text(payload: FormalizeRequest):
    input_text = payload.text.strip()
    if not input_text:
        return {"formalized": ""}
    refined = gemini_service.formalize_prompt(input_text)
    return {"original": input_text, "formalized": refined}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "app_name": "OmniMind AI",
        "api_configured": gemini_service.is_configured(),
        "default_model": gemini_service.primary_model,
        "available_models": [config.GEMINI_MODEL] + config.FALLBACK_MODELS
    }

@app.get("/api/quick-actions")
async def get_quick_actions():
    return QUICK_ACTION_CARDS

@app.get("/api/models")
async def get_models():
    return {
        "current": gemini_service.primary_model,
        "options": [
            {"id": "gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash Lite (Recommended)", "desc": "Fast, high-fidelity reasoning, generous free tier quota"},
            {"id": "gemini-3.8-flash", "name": "Gemini 3.8 Flash", "desc": "Deep reasoning & multi-step problem solving"},
            {"id": "gemini-3.6-flash", "name": "Gemini 3.6 Flash", "desc": "Multimodal, balanced performance"}
        ]
    }

@app.get("/api/conversations")
async def list_conversations(q: Optional[str] = Query(None, description="Search query")):
    conversations = get_conversations(search_query=q)
    return {"conversations": conversations}

@app.post("/api/conversations", status_code=status.HTTP_201_CREATED)
async def new_conversation(category: str = "general"):
    conv = create_conversation(title="New Chat", category=category)
    return conv

@app.get("/api/conversations/{conv_id}")
async def fetch_conversation(conv_id: str):
    conv = get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv

@app.patch("/api/conversations/{conv_id}")
async def rename_conversation(conv_id: str, payload: TitleUpdateRequest):
    success = update_conversation_title(conv_id, payload.title.strip())
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True, "title": payload.title.strip()}

@app.delete("/api/conversations/{conv_id}")
async def remove_conversation(conv_id: str):
    success = delete_conversation(conv_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True, "id": conv_id}

@app.post("/api/conversations/{conv_id}/clear")
async def clear_messages(conv_id: str):
    clear_conversation_messages(conv_id)
    return {"success": True, "conversation_id": conv_id}

@app.post("/api/settings/key")
async def set_api_key(payload: ApiKeyUpdateRequest):
    gemini_service.update_api_key(payload.api_key.strip())
    return {
        "success": True,
        "configured": gemini_service.is_configured(),
        "message": "Gemini API key updated successfully."
    }

@app.post("/api/chat")
async def chat_non_streaming(req: ChatRequest):
    message_text = req.message.strip()
    if not message_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty or whitespace.")

    # Get or create conversation
    conv_id = req.conversation_id
    is_new = False
    if not conv_id:
        new_conv = create_conversation(title=auto_generate_title(message_text), category=req.category or "general")
        conv_id = new_conv["id"]
        is_new = True
    else:
        existing = get_conversation(conv_id)
        if not existing:
            new_conv = create_conversation(title=auto_generate_title(message_text), category=req.category or "general")
            conv_id = new_conv["id"]
            is_new = True
        elif existing["title"] == "New Chat":
            update_conversation_title(conv_id, auto_generate_title(message_text))

    # Retrieve previous conversation context
    history = prepare_conversation_context(conv_id, max_messages=12)

    # Persist user message
    user_msg = add_message(conv_id=conv_id, role="user", content=message_text)

    try:
        res = gemini_service.generate_chat_response(
            history=history,
            message=message_text,
            style=req.style,
            language=req.language,
            custom_instructions=req.custom_instructions,
            preferred_model=req.model
        )
        assistant_msg = add_message(
            conv_id=conv_id,
            role="assistant",
            content=res["content"],
            model=res["model"],
            suggested_followups=res["suggested_followups"]
        )
        return {
            "conversation_id": conv_id,
            "user_message": user_msg,
            "assistant_message": assistant_msg
        }
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

@app.post("/api/chat/stream")
async def chat_streaming(req: ChatRequest):
    message_text = req.message.strip()
    if not message_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty or whitespace.")

    # Get or create conversation
    conv_id = req.conversation_id
    if not conv_id:
        new_conv = create_conversation(title=auto_generate_title(message_text), category=req.category or "general")
        conv_id = new_conv["id"]
    else:
        existing = get_conversation(conv_id)
        if not existing:
            new_conv = create_conversation(title=auto_generate_title(message_text), category=req.category or "general")
            conv_id = new_conv["id"]
        elif existing["title"] == "New Chat":
            update_conversation_title(conv_id, auto_generate_title(message_text))

    # Context history before saving current user message
    history = prepare_conversation_context(conv_id, max_messages=12)

    # Persist user message in SQLite
    user_msg = add_message(conv_id=conv_id, role="user", content=message_text)

    def event_stream():
        # First send handshake event with conversation_id and user message info
        yield f"data: {json.dumps({'type': 'init', 'conversation_id': conv_id, 'user_message': user_msg})}\n\n"

        full_content = ""
        model_used = req.model or gemini_service.primary_model
        followups = []

        try:
            for item in gemini_service.stream_chat_response(
                history=history,
                message=message_text,
                style=req.style,
                language=req.language,
                custom_instructions=req.custom_instructions,
                preferred_model=req.model
            ):
                if item["type"] == "token":
                    full_content += item["content"]
                    yield f"data: {json.dumps(item)}\n\n"
                elif item["type"] == "done":
                    full_content = item["content"]
                    followups = item.get("suggested_followups", [])
                    model_used = item.get("model", model_used)
                elif item["type"] == "error":
                    yield f"data: {json.dumps(item)}\n\n"
                    return

            # Save the complete assistant response to SQLite
            saved_msg = add_message(
                conv_id=conv_id,
                role="assistant",
                content=full_content,
                model=model_used,
                suggested_followups=followups
            )
            yield f"data: {json.dumps({'type': 'complete', 'message': saved_msg})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/conversations/{conv_id}/regenerate")
async def regenerate_response(conv_id: str, style: Optional[str] = "default", language: Optional[str] = "English"):
    conv = get_conversation(conv_id)
    if not conv or not conv.get("messages"):
        raise HTTPException(status_code=404, detail="No messages to regenerate")

    # Delete last assistant message and retrieve last prompt
    last_prompt = delete_last_assistant_message(conv_id)
    if not last_prompt:
        raise HTTPException(status_code=400, detail="Cannot find preceding user prompt to regenerate.")

    # Context history
    history = prepare_conversation_context(conv_id, max_messages=12)

    def event_stream():
        full_content = ""
        model_used = gemini_service.primary_model
        followups = []

        try:
            for item in gemini_service.stream_chat_response(
                history=history,
                message=last_prompt,
                style=style,
                language=language
            ):
                if item["type"] == "token":
                    full_content += item["content"]
                    yield f"data: {json.dumps(item)}\n\n"
                elif item["type"] == "done":
                    full_content = item["content"]
                    followups = item.get("suggested_followups", [])
                    model_used = item.get("model", model_used)
                elif item["type"] == "error":
                    yield f"data: {json.dumps(item)}\n\n"
                    return

            saved_msg = add_message(
                conv_id=conv_id,
                role="assistant",
                content=full_content,
                model=model_used,
                suggested_followups=followups
            )
            yield f"data: {json.dumps({'type': 'complete', 'message': saved_msg})}\n\n"
        except Exception as e:
            logger.error(f"Regenerate error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# Mount Frontend Static Build if present (enables standalone single-port serving)
possible_dist_dirs = [
    config.BASE_DIR.parent / "frontend" / "dist",
    config.BASE_DIR / "frontend" / "dist",
    config.BASE_DIR / "dist",
]
FRONTEND_DIST = None
for d in possible_dist_dirs:
    if d.exists() and (d / "index.html").exists():
        FRONTEND_DIST = d
        break

if FRONTEND_DIST:
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    logger.info(f"Mounted frontend dist from {FRONTEND_DIST}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=True)
