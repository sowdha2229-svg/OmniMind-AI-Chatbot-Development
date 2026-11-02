# OmniMind AI – Your Intelligent AI Companion

> **MSc Data Science Academic Project**  
> *Universal, Context-Aware Intelligent AI Assistant powered by Google Gemini and Modern Full-Stack Architecture.*

---

## 1. Executive Summary

**OmniMind AI** is a state-of-the-art, general-purpose intelligent AI companion developed for an MSc Data Science academic initiative. Moving far beyond trivial chatbot wrappers, OmniMind AI combines the reasoning capabilities of **Google Gemini** with a high-performance **FastAPI** backend, persistent **SQLite** conversation storage, real-time **Server-Sent Events (SSE)** token streaming, and a responsive **React (Vite + Tailwind CSS)** user interface.

It features complete multi-turn conversational memory, mathematical derivation rendering using **LaTeX (KaTeX)**, syntax-highlighted code execution blocks with one-click clipboard copying, dynamic follow-up suggestions, and specialized assistance spanning Data Science, Programming, Mathematics, Research, and Creative Problem Solving.

---

## 2. System Architecture

```
[ Client Layer (Vite + React 18 + Tailwind) ]
    │
    ├── Server-Sent Events (SSE) Streaming Interface
    ├── Markdown & KaTeX LaTeX Equation Renderer
    ├── Dynamic Follow-Up Question Chips
    └── Theme Engine (Light / Dark Mode)
          │
          ▼ HTTP / SSE Proxy (Port 5173 -> 8000)
[ API Gateway Layer (FastAPI + Uvicorn) ]
    │
    ├── /api/chat/stream  -> Real-Time SSE Token Stream
    ├── /api/chat         -> Synchronous Turn Execution
    ├── /api/conversations-> CRUD Session Management & Search
    └── /api/settings     -> Dynamic Model & Key Tuning
          │
    ┌─────┴─────────────────────────────────┐
    ▼                                       ▼
[ Conversation & Persistence Engine ]    [ Google Gen AI Service ]
    │                                       │
    ├── SQLite Storage (omnimind.db)        ├── Official `google-genai` SDK
    ├── Multi-Turn Context Pruning          ├── Gemini 3.6 / 3.8 Flash
    └── Dynamic Auto-Titling Heuristics     ├── IPv4 Safe Socket Transport
                                            └── System Directive Engine
```

---

## 3. Key Features & Capabilities

### A. Universal AI Assistant
- Cross-domain assistance across Data Science, Artificial Intelligence, Mathematics, Science, Technology, Business, Career, Language, and Daily Problem Solving.
- Translates content fluently across languages (English, Tamil, Spanish, French, German, Hindi, Japanese, Arabic).
- Handles multi-step academic reasoning and structured markdown output.

### B. Advanced Programming Assistant
- Generates complete, runnable, production-ready code in Python, Java, C++, JavaScript, TypeScript, Rust, and SQL.
- Provides line-by-line explanations, identifies edge cases, optimizes algorithms, and debugs logic errors.
- Deep integration with Python data libraries: **NumPy**, **Pandas**, **Scikit-Learn**, **Matplotlib**, **Seaborn**, **TensorFlow**, and **PyTorch**.

### C. Intelligent Learning & Mathematics Assistant
- Step-by-step mathematical problem solving and theoretical derivations.
- High-fidelity **KaTeX LaTeX math rendering** for inline equations (`$...$`) and block equations (`$$...$$`).
- Generates study outlines, exam preparation materials, and multiple-choice questions (MCQs).

### D. Advanced Data Science Assistant
- Complete end-to-end data pipelines: Missing value imputation, Outlier detection, Exploratory Data Analysis (EDA), Feature Engineering, Normalization, SMOTE, Model Training, Cross-Validation, and ROC-AUC / Confusion Matrix evaluation.

### E. Smart Conversation Memory & Multi-Turn Context
- Persistent context awareness across conversational turns (e.g. referencing "its" refers to the concept established in previous turns).
- Retains up to 12 recent message exchanges dynamically formatted into Gemini Content roles.

### F. High-Performance Streaming & Interaction
- Real-time **Server-Sent Events (SSE)** token streaming with smooth typing animation.
- Instant **Stop Generation** control via browser `AbortController`.
- **Regenerate Response** functionality for assistant turns.
- One-click **Copy Response** and **Copy Code** with visual confirmation checkmarks.
- Interactive **Suggested Follow-Up Questions** chips generated dynamically by Gemini.

### G. Professional & Formal Language Enforcement
- **MSc Academic Formality Standard**: If a user submits informal words, street slang, texting shorthand (e.g. *yo, bro, gimme, wazzup, idk, lol, wat is dis*), or ungrammatical slang, OmniMind AI politely guides the user to reformulate the request in a formal and professional manner.
- **Interactive Rephrase Chips**: The suggested professional rephrasing is generated directly into the response and interactive chips, enabling one-click submission of the refined inquiry.
- **Wand Formalize Tool**: A built-in `Formalize` helper beside the chat input bar dynamically rewrites informal drafts into academic-grade prompts on demand (`/api/formalize`).

---

## 4. Worldwide Single-Link Deployment (Cloudflare Tunnel)

OmniMind AI is configured with instant public tunneling via **Cloudflare Quick Tunnels**. This provides a single, secure HTTPS link accessible from **any device (mobile phones, laptops, tablets, external networks)** without requiring logins, passwords, or port forwarding.

### Live Universal Public Link
> 🌐 **`https://practitioner-presenting-qualification-exclusion.trycloudflare.com`**

To launch or re-host the public tunnel anytime on Windows:
```cmd
share_public_url.bat
```

---

## 4. Project File Structure

```text
OmniMind-AI/
├── backend/
│   ├── app.py                  # FastAPI application, routes, and static mount
│   ├── gemini_service.py       # Google Gen AI SDK integration & system prompts
│   ├── conversation.py         # Multi-turn context management & quick actions
│   ├── database.py             # SQLite persistence schema & CRUD operations
│   ├── config.py               # Environment configuration & IPv4 network safety
│   ├── test_suite.py           # Comprehensive 22-step automated verification suite
│   ├── requirements.txt        # Python backend dependencies
│   ├── .env                    # Runtime environment secrets
│   └── .env.example            # Environment template
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatArea.jsx       # Chat feed, message stream & input box
│   │   │   ├── CodeBlock.jsx      # Syntax highlighted code with copy button
│   │   │   ├── Header.jsx         # Controls, style & language selectors
│   │   │   ├── MessageBubble.jsx  # Markdown, KaTeX, avatars, chips & toolbar
│   │   │   ├── SettingsModal.jsx  # AI engine selection & custom persona
│   │   │   ├── Sidebar.jsx        # Navigation, search, rename & delete
│   │   │   └── WelcomeScreen.jsx  # Hero section & 8 Quick Action Cards
│   │   ├── services/
│   │   │   └── api.js             # SSE stream reader and REST client
│   │   ├── styles/
│   │   │   └── index.css          # Tailwind directives, fonts & glassmorphism
│   │   ├── App.jsx                # Application root state coordinator
│   │   └── main.jsx               # React entry point
│   ├── dist/                      # Pre-compiled production build
│   ├── index.html                 # HTML shell with KaTeX and Google Fonts
│   ├── package.json               # Frontend dependencies and scripts
│   ├── vite.config.js             # Vite configuration with API proxy
│   ├── tailwind.config.js         # Custom palette and animations
│   └── postcss.config.js          # PostCSS configuration
├── run_backend.bat             # 1-Click Windows Backend Launcher
├── run_frontend.bat            # 1-Click Windows Frontend Launcher
├── start_all.bat               # 1-Click Master Application Launcher
└── README.md                   # Complete academic documentation
```

---

## 5. Quickstart & Installation

### Prerequisites
- **Python**: 3.10+ (Tested on Python 3.14 x64)
- **Node.js**: 18+ (Tested on Node.js v26.13)
- **Google Gemini API Key**: Obtainable for free at [Google AI Studio](https://aistudio.google.com/)

---

### Step 1: Configure Environment Variables

Navigate to `backend/` and configure `.env`:

```bash
# backend/.env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
HOST=0.0.0.0
PORT=8000
```

---

### Step 2: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

---

### Step 3: Launch OmniMind AI

#### Option A: One-Click Master Launcher (Windows)
Double-click:
```text
start_all.bat
```
This automatically starts both the FastAPI backend and Vite frontend dev server, and opens your default browser at `http://localhost:5173`.

#### Option B: Standalone Production Mode (Single Port)
The FastAPI backend serves the pre-compiled React application directly from `frontend/dist`:
```bash
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in any browser.

#### Option C: Development Mode (Hot Reloading)
In Terminal 1 (Backend):
```bash
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

In Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```
Open **`http://localhost:5173`**.

---

## 6. Automated Verification Test Suite

A comprehensive 22-step automated test suite is included in `backend/test_suite.py` covering all core functionalities:

```bash
cd backend
python test_suite.py
```

### Verification Output:
```text
=================================================================
       OMNIMIND AI - FULL VERIFICATION TEST SUITE
=================================================================

--- Phase 1: Service Health & Config ---
 [PASS] GET /api/health returns 200 
 [PASS] Gemini API is configured 
 [PASS] Default model present 
 [PASS] GET /api/quick-actions returns 200 
 [PASS] 8 Quick Action cards present 
 [PASS] GET /api/models returns 200 
 [PASS] Model options list populated 

--- Phase 2: Conversation Session Management ---
 [PASS] POST /api/conversations creates session 
 [PASS] Conversation ID generated 
 [PASS] PATCH /api/conversations/{id} renames title 
 [PASS] GET /api/conversations with search query returns matching 

--- Phase 3: Validation & Error Handling ---
 [PASS] POST /api/chat rejects empty / whitespace message with 400 

--- Phase 4: Gemini AI Generation & Reasoning ---
 -> Prompt: Explain Principal Component Analysis (PCA) in two sentences with its mathematical objective.
 [PASS] POST /api/chat generates response (200) 
 [PASS] AI generated substantive explanation (595 chars)
 [PASS] Followup questions generated 

--- Phase 5: Multi-Turn Memory & Context Continuity ---
 -> Follow-up prompt: What is the primary formula used to find its eigenvalues and eigenvectors?
 [PASS] Multi-turn follow-up answered successfully 
 [PASS] Contextual understanding preserved ('its' -> PCA / covariance matrix) 

--- Phase 6: Session Cleanup & Persistence ---
 [PASS] POST /api/conversations/{id}/clear clears messages 
 [PASS] Conversation messages array is now empty 
 [PASS] DELETE /api/conversations/{id} deletes conversation 
 [PASS] Deleted conversation returns 404 
 [PASS] Root GET / serves React frontend dist index.html 

=================================================================
 TEST RESULTS SUMMARY: 22 PASSED, 0 FAILED
=================================================================
```

---

## 7. Interactive API Documentation

Interactive Swagger documentation is automatically generated by FastAPI:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc UI**: `http://localhost:8000/redoc`

---

## 8. License & Academic Attribution
Developed as part of the **MSc Data Science Program**. Powered by **Google Gemini AI**. All code is structured for academic presentation, production deployment, and real-world AI exploration.
