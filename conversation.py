import re
import logging
from typing import List, Dict, Any, Optional
from database import (
    get_conversation,
    get_conversations,
    create_conversation,
    update_conversation_title,
    delete_conversation,
    clear_conversation_messages,
    add_message,
    delete_last_assistant_message
)
from gemini_service import gemini_service

logger = logging.getLogger("omnimind.conversation")

# 8 Quick Action card definitions required by prompt
QUICK_ACTION_CARDS = [
    {
        "id": "ask-anything",
        "title": "Ask Anything",
        "icon": "HelpCircle",
        "description": "Universal reasoning across science, technology, arts, history & daily life",
        "prompt": "Explain the Fermi Paradox and the most compelling scientific hypotheses proposed to resolve it."
    },
    {
        "id": "generate-code",
        "title": "Generate Code",
        "icon": "Code",
        "description": "Production-grade code in Python, Java, C++, JS, Rust & SQL with debugging",
        "prompt": "Write a complete, robust Python implementation of an LRU (Least Recently Used) Cache using an OrderedDict and Doubly Linked List with full type hints and unit tests."
    },
    {
        "id": "explain-concepts",
        "title": "Explain Concepts",
        "icon": "BookOpen",
        "description": "Step-by-step academic explanations, intuitive analogies and clear breakdowns",
        "prompt": "Explain Quantum Entanglement and Bell's Theorem step-by-step for a curious graduate student."
    },
    {
        "id": "data-science",
        "title": "Data Science Assistant",
        "icon": "Database",
        "description": "End-to-end data preprocessing, EDA, feature engineering, ML models & evaluation",
        "prompt": "Provide a complete end-to-end Python pipeline for customer churn prediction using Scikit-Learn: include data cleaning, feature engineering, SMOTE for imbalance, XGBoost training, and ROC-AUC / Confusion Matrix evaluation."
    },
    {
        "id": "solve-math",
        "title": "Solve Mathematics",
        "icon": "Percent",
        "description": "Mathematical, statistical & algorithmic problems with complete LaTeX steps",
        "prompt": "Derive the closed-form Ordinary Least Squares (OLS) estimator $\\hat{\\beta} = (X^T X)^{-1} X^T y$ step-by-step from first principles using matrix calculus."
    },
    {
        "id": "write-content",
        "title": "Write Content",
        "icon": "FileText",
        "description": "Executive reports, research abstracts, proposals, emails & formal documentation",
        "prompt": "Draft an executive research proposal abstract for an MSc Data Science dissertation on 'Self-Supervised Contrastive Learning for Multi-Modal Medical Image Diagnostics'."
    },
    {
        "id": "career-assistant",
        "title": "Career Assistant",
        "icon": "Briefcase",
        "description": "Technical interview coaching, resume optimization, career roadmaps & mock prep",
        "prompt": "Simulate a mock technical interview for a Senior Machine Learning Engineer position. Ask me three challenging questions on Transformer architecture and attention mechanisms."
    },
    {
        "id": "creative-ideas",
        "title": "Creative Ideas",
        "icon": "Sparkles",
        "description": "Innovative brainstorming, product ideation, narratives, and creative problem solving",
        "prompt": "Brainstorm 5 innovative, high-impact AI startup concepts combining Large Multimodal Models with autonomous edge robotics for precision agriculture."
    }
]

def auto_generate_title(first_prompt: str) -> str:
    """Generate a clean, punchy 3-6 word title from the user prompt."""
    if not first_prompt:
        return "New Chat"
    
    # Try fast heuristic first
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', first_prompt).strip()
    words = cleaned.split()
    if len(words) <= 5 and len(cleaned) <= 35:
        return " ".join(words).capitalize()
    
    # Quick fallback summarize
    candidate = " ".join(words[:5]).capitalize()
    if candidate:
        return candidate
    return "OmniMind Discussion"

def prepare_conversation_context(conv_id: str, max_messages: int = 14) -> List[Dict[str, str]]:
    """Retrieve and format prior conversation history for multi-turn context."""
    conv = get_conversation(conv_id)
    if not conv:
        return []
    
    all_msgs = conv.get("messages", [])
    # Take the latest N messages to maintain high coherence while preserving token headroom
    recent_msgs = all_msgs[-max_messages:]
    
    history = []
    for m in recent_msgs:
        history.append({
            "role": m["role"],
            "content": m["content"]
        })
    return history
