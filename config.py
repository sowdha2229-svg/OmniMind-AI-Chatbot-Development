import os
import sys
import socket
from pathlib import Path
from dotenv import load_dotenv

# Apply IPv4-only patch on Windows to fix SSL/EOF errors with Google APIs
# On Linux (Railway, Render, cloud servers) this patch is NOT needed and skipped
if sys.platform == "win32":
    orig_getaddrinfo = socket.getaddrinfo
    def ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
    socket.getaddrinfo = ipv4_getaddrinfo

# Locate and load .env file
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# Configuration values
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
FALLBACK_MODELS = ["gemini-3.1-flash-lite-preview", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DATABASE_PATH = BASE_DIR / "omnimind.db"
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"
]
