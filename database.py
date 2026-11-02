import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from config import DATABASE_PATH

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            model TEXT,
            suggested_followups TEXT, -- JSON string list
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC);")
    conn.commit()
    conn.close()

def create_conversation(title: str = "New Chat", category: str = "general") -> Dict[str, Any]:
    conv_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (id, title, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (conv_id, title, category, now, now)
    )
    conn.commit()
    conn.close()
    return {
        "id": conv_id,
        "title": title,
        "category": category,
        "created_at": now,
        "updated_at": now,
        "message_count": 0
    }

def get_conversations(search_query: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    if search_query:
        query = """
            SELECT c.*, COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE c.title LIKE ? OR m.content LIKE ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """
        param = f"%{search_query.strip()}%"
        cursor.execute(query, (param, param))
    else:
        query = """
            SELECT c.*, COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """
        cursor.execute(query)
    
    rows = cursor.fetchall()
    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "title": r["title"],
            "category": r["category"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "message_count": r["message_count"]
        })
    conn.close()
    return result

def get_conversation(conv_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
    conv_row = cursor.fetchone()
    if not conv_row:
        conn.close()
        return None
    
    cursor.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conv_id,)
    )
    msg_rows = cursor.fetchall()
    messages = []
    for m in msg_rows:
        followups = []
        if m["suggested_followups"]:
            try:
                followups = json.loads(m["suggested_followups"])
            except Exception:
                followups = []
        messages.append({
            "id": m["id"],
            "role": m["role"],
            "content": m["content"],
            "model": m["model"],
            "suggested_followups": followups,
            "created_at": m["created_at"]
        })
    
    conn.close()
    return {
        "id": conv_row["id"],
        "title": conv_row["title"],
        "category": conv_row["category"],
        "created_at": conv_row["created_at"],
        "updated_at": conv_row["updated_at"],
        "messages": messages
    }

def update_conversation_title(conv_id: str, new_title: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    cursor.execute(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
        (new_title, now, conv_id)
    )
    affected = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return affected

def delete_conversation(conv_id: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    affected = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return affected

def clear_conversation_messages(conv_id: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    now = datetime.utcnow().isoformat()
    cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
    conn.commit()
    conn.close()
    return True

def add_message(conv_id: str, role: str, content: str, model: Optional[str] = None, suggested_followups: Optional[List[str]] = None) -> Dict[str, Any]:
    msg_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    followups_json = json.dumps(suggested_followups) if suggested_followups else None
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO messages (id, conversation_id, role, content, model, suggested_followups, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (msg_id, conv_id, role, content, model, followups_json, now)
    )
    cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
    conn.commit()
    conn.close()
    return {
        "id": msg_id,
        "conversation_id": conv_id,
        "role": role,
        "content": content,
        "model": model,
        "suggested_followups": suggested_followups or [],
        "created_at": now
    }

def delete_last_assistant_message(conv_id: str) -> Optional[str]:
    """Used for regenerate functionality: deletes the most recent assistant message and returns the last user prompt."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, role, content FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 2",
        (conv_id,)
    )
    rows = cursor.fetchall()
    if not rows:
        conn.close()
        return None
    
    if rows[0]["role"] == "assistant":
        cursor.execute("DELETE FROM messages WHERE id = ?", (rows[0]["id"],))
        conn.commit()
        last_user_prompt = rows[1]["content"] if len(rows) > 1 and rows[1]["role"] == "user" else None
        conn.close()
        return last_user_prompt
    
    conn.close()
    return rows[0]["content"] if rows[0]["role"] == "user" else None
