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
    config.BASE_DIR / "dist",
    config.BASE_DIR,
    config.BASE_DIR.parent / "frontend" / "dist",
    config.BASE_DIR / "frontend" / "dist",
    Path.cwd() / "dist",
    Path.cwd(),
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
else:
    from fastapi.responses import HTMLResponse

    BUILTIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OmniMind AI — Universal Intelligent Companion</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        pre, code { font-family: 'Fira Code', monospace; }
        .prose pre { background: #1e293b; color: #f8fafc; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; margin: 0.5rem 0; }
        .prose code { background: #e2e8f0; color: #0f172a; padding: 0.15rem 0.35rem; border-radius: 0.25rem; font-size: 0.875rem; }
        .dark .prose code { background: #334155; color: #f8fafc; }
        .dot-flashing { animation: dotFlashing 1s infinite alternate; }
        @keyframes dotFlashing { 0% { opacity: 0.2; } 100% { opacity: 1; } }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col">
    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-20 px-4 py-3 flex items-center justify-between">
        <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center font-bold text-lg shadow-lg shadow-indigo-500/20">
                ✦
            </div>
            <div>
                <h1 class="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">OmniMind AI</h1>
                <p class="text-xs text-slate-400">Universal Intelligent Companion</p>
            </div>
        </div>
        <div class="flex items-center gap-2">
            <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> Online
            </span>
            <button onclick="newChat()" class="px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 transition">
                + New Chat
            </button>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-1 max-w-4xl w-full mx-auto p-4 flex flex-col">
        <!-- Messages Area -->
        <div id="messages" class="flex-1 overflow-y-auto space-y-4 mb-4 pr-1">
            <!-- Welcome Hero -->
            <div id="welcome" class="text-center py-12 px-4 space-y-4">
                <div class="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-tr from-indigo-600 to-purple-600 flex items-center justify-center text-3xl shadow-xl shadow-indigo-500/20">
                    ✦
                </div>
                <h2 class="text-2xl font-bold text-white">How can OmniMind help you today?</h2>
                <p class="text-slate-400 text-sm max-w-md mx-auto">Ask questions, brainstorm concepts, debug code, or solve complex problems with AI.</p>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg mx-auto pt-4 text-left">
                    <button onclick="sendPrompt('Explain quantum computing in simple terms')" class="p-3 rounded-xl bg-slate-800/80 hover:bg-slate-800 border border-slate-700/60 hover:border-indigo-500/50 text-xs text-slate-300 transition">
                        💡 <strong class="text-white block">Explain a concept</strong> Quantum computing in simple terms
                    </button>
                    <button onclick="sendPrompt('Write a clean Python function for binary search with comments')" class="p-3 rounded-xl bg-slate-800/80 hover:bg-slate-800 border border-slate-700/60 hover:border-indigo-500/50 text-xs text-slate-300 transition">
                        ⚡ <strong class="text-white block">Code assistant</strong> Python binary search algorithm
                    </button>
                    <button onclick="sendPrompt('Give me 5 productivity habits backed by science')" class="p-3 rounded-xl bg-slate-800/80 hover:bg-slate-800 border border-slate-700/60 hover:border-indigo-500/50 text-xs text-slate-300 transition">
                        🎯 <strong class="text-white block">Productivity</strong> 5 science-backed daily habits
                    </button>
                    <button onclick="sendPrompt('What are the key differences between SQL and NoSQL databases?')" class="p-3 rounded-xl bg-slate-800/80 hover:bg-slate-800 border border-slate-700/60 hover:border-indigo-500/50 text-xs text-slate-300 transition">
                        📊 <strong class="text-white block">System Design</strong> SQL vs NoSQL comparison
                    </button>
                </div>
            </div>
        </div>

        <!-- Input Box -->
        <div class="sticky bottom-4 bg-slate-900/90 backdrop-blur border border-slate-700/80 rounded-2xl p-3 shadow-2xl">
            <div class="flex items-center gap-2 mb-2 text-xs text-slate-400 px-1">
                <span>Style:</span>
                <select id="styleSelect" class="bg-slate-800 border border-slate-700 rounded px-2 py-0.5 text-xs text-slate-200 outline-none">
                    <option value="default">Default</option>
                    <option value="concise">Concise & Direct</option>
                    <option value="eli5">Explain Like I'm 5</option>
                    <option value="deep_dive">Deep Dive Technical</option>
                </select>
            </div>
            <div class="flex items-end gap-2">
                <textarea id="userInput" rows="1" placeholder="Type your message... (Enter to send, Shift+Enter for new line)" 
                    class="flex-1 bg-transparent border-0 text-slate-100 placeholder-slate-500 resize-none outline-none text-sm p-1 max-h-36"
                    onkeydown="handleKey(event)" oninput="autoGrow(this)"></textarea>
                <button id="sendBtn" onclick="sendMessage()" class="px-4 py-2 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-medium text-xs rounded-xl shadow-lg shadow-indigo-500/25 transition disabled:opacity-50">
                    Send ➔
                </button>
            </div>
        </div>
    </main>

    <script>
        let currentConvId = null;
        let isGenerating = false;

        function autoGrow(el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 144) + 'px';
        }

        function handleKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        function sendPrompt(text) {
            document.getElementById('userInput').value = text;
            sendMessage();
        }

        function newChat() {
            currentConvId = null;
            document.getElementById('messages').innerHTML = '';
            document.getElementById('welcome')?.classList.remove('hidden');
        }

        async function sendMessage() {
            if (isGenerating) return;
            const input = document.getElementById('userInput');
            const text = input.value.trim();
            if (!text) return;

            // Hide welcome
            document.getElementById('welcome')?.classList.add('hidden');

            // Append User Message
            appendUserMessage(text);
            input.value = '';
            input.style.height = 'auto';

            // Placeholder Assistant Message
            const assistantBubble = appendAssistantPlaceholder();
            isGenerating = true;
            document.getElementById('sendBtn').disabled = true;

            try {
                const response = await fetch('/api/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: text,
                        conversation_id: currentConvId,
                        style: document.getElementById('styleSelect').value
                    })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'API request failed');
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let accumulated = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.type === 'init') {
                                    currentConvId = data.conversation_id;
                                } else if (data.type === 'token') {
                                    accumulated += data.content;
                                    assistantBubble.innerHTML = marked.parse(accumulated);
                                    window.scrollTo(0, document.body.scrollHeight);
                                } else if (data.type === 'error') {
                                    assistantBubble.innerHTML = '<span class="text-rose-400">⚠ ' + data.error + '</span>';
                                }
                            } catch (e) {}
                        }
                    }
                }
            } catch (err) {
                assistantBubble.innerHTML = '<span class="text-rose-400">⚠ Error: ' + err.message + '</span>';
            } finally {
                isGenerating = false;
                document.getElementById('sendBtn').disabled = false;
            }
        }

        function appendUserMessage(text) {
            const container = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = 'flex justify-end';
            div.innerHTML = '<div class="max-w-[85%] bg-indigo-600 text-white px-4 py-2.5 rounded-2xl rounded-tr-sm text-sm shadow-md">' + escapeHtml(text) + '</div>';
            container.appendChild(div);
            window.scrollTo(0, document.body.scrollHeight);
        }

        function appendAssistantPlaceholder() {
            const container = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = 'flex justify-start gap-3';
            div.innerHTML = `
                <div class="w-8 h-8 rounded-lg bg-purple-600/30 border border-purple-500/30 flex items-center justify-center text-xs font-bold text-purple-300 shrink-0">✦</div>
                <div class="max-w-[85%] bg-slate-800/80 border border-slate-700/60 text-slate-100 px-4 py-3 rounded-2xl rounded-tl-sm text-sm prose prose-invert overflow-hidden">
                    <span class="dot-flashing text-slate-400">Thinking...</span>
                </div>
            `;
            container.appendChild(div);
            window.scrollTo(0, document.body.scrollHeight);
            return div.querySelector('.prose');
        }

        function escapeHtml(str) {
            return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }
    </script>
</body>
</html>
"""

    @app.get("/", response_class=HTMLResponse)
    async def serve_builtin_ui():
        return BUILTIN_HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=True)
