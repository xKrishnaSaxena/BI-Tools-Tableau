import json
import os
import re
import csv
import uuid
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field,field_validator
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# ----------------- Logging -----------------
def _setup_logging():
    level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.DEBUG),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for noisy in ("httpx", "urllib3", "langchain", "google", "fastapi", "uvicorn"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

_setup_logging()
log = logging.getLogger("tableau-agent")
load_dotenv()

# ---------------- Optional LLM backends ----------------
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except Exception as e:
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None
    log.warning("langchain_google_genai not available: %s", e)

try:
    from langchain_openai import ChatOpenAI  # fallback only if Gemini unavailable
except Exception as e:
    ChatOpenAI = None
    log.warning("langchain_openai not available: %s", e)

# ---------------- Tableau tool shims ----------------
from tab_tools import (
    list_tableau_projects,
    list_tableau_workbooks,
    list_tableau_views,
    tableau_get_view_image,
    tableau_get_view_data,
    publish_mock_datasource,
)

# ---------------- LangMem (semantic memory) ----------------
try:
    from langmem import create_memory_store_manager  # type: ignore
    from langgraph.store.memory import InMemoryStore  # type: ignore
except Exception as e:
    create_memory_store_manager = None  # type: ignore
    InMemoryStore = None  # type: ignore
    log.warning("LangMem modules not available: %s", e)

# ---------------- App setup ----------------
app = FastAPI(
    title="Tableau Chatbot Agent API",
    description="Fast Tableau tool-caller with schema-free long-term memory (LangMem + user-tail).",
    version="2.7.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://agent-tableau.onrender.com",
        "http://localhost:5173",
        "https://enmployee-os-gq96.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LLM selection ----------------
def make_llm():
    if ChatGoogleGenerativeAI is not None and os.getenv("GOOGLE_API_KEY"):
        try:
            log.info("Using Gemini chat model: gemini-2.0-flash-lite")
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
        except Exception as e:
            log.error("Failed to init Gemini chat model: %s", e, exc_info=True)
    if ChatOpenAI is not None and os.getenv("OPENAI_API_KEY"):
        try:
            log.info("Falling back to OpenAI chat model: gpt-4o-mini")
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception as e:
            log.error("Failed to init OpenAI chat model: %s", e, exc_info=True)
    raise RuntimeError("No LLM configured. Set GOOGLE_API_KEY (Gemini) or OPENAI_API_KEY (OpenAI).")

llm = make_llm()


# ---------------- Tools wiring ----------------
TOOLS = [
    list_tableau_projects,
    list_tableau_workbooks,
    list_tableau_views,
    tableau_get_view_image,
    tableau_get_view_data,
    publish_mock_datasource,
]
TOOL_REGISTRY = {t.name: t for t in TOOLS}

# ---- Pending-intent (per-session) ----
PENDING: Dict[str, Dict[str, Any]] = {}

def get_pending(session_id: str) -> Optional[Dict[str, Any]]:
    return PENDING.get(session_id)

def set_pending(session_id: str, intent: str, data: Optional[Dict[str, Any]] = None) -> None:
    log.debug("Set pending intent: session=%s intent=%s data=%s", session_id, intent, data)
    PENDING[session_id] = {"intent": intent, "data": data or {}}

def clear_pending(session_id: str) -> None:
    if session_id in PENDING:
        log.debug("Clear pending intent: session=%s", session_id)
        del PENDING[session_id]

# ---------------- LLM intent & slot schema ----------------
class Intent(str, Enum):
    VIEW_IMAGE = "view_image"               # show a Tableau view image
    VIEW_DATA = "view_data"                 # analyze/export data from a Tableau view
    PUBLISH_MOCK = "publish_mock_datasource"
    MEMORY_QA = "memory_qa"                 # "what's my ...", "do you remember ..."
    HELP = "help"                           # "what can you do", "help"
    SMALLTALK = "smalltalk"                 # chit-chat
    UNKNOWN = "unknown"

class PublishArgs(BaseModel):
    project_name: Optional[str] = None
    datasource_name: Optional[str] = None
    overwrite: Optional[bool] = None

class ParsedCommand(BaseModel):
    intent: Intent = Field(..., description="Best intent for this user message.")
    # tableau view access
    view_name: Optional[str] = None
    workbook_name: Optional[str] = None
    filters_json: Optional[Dict[str, Any]] = None   
    # Accept '{"k":"v"}' as well as dict, and coerce to dict.
    @field_validator("filters_json", mode="before")
    @classmethod
    def _filters_json_coerce(cls, v):
        if v is None:
            return None
        if isinstance(v, (dict,)):
            return v
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None
    publish: Optional[PublishArgs] = None
    # memory
    is_specific_memory_question: Optional[bool] = False
    # misc
    reason: str = Field(..., description="Short rationale of why this intent/slots were chosen.")

def _llm_parse_user_utterance(user_text: str, recalled_snippets: List[str]) -> ParsedCommand:
    """
    Prompt-based router+slot extractor that returns a ParsedCommand via structured output.
    Uses whatever LLM you've already configured in `make_llm()`.
    """
    # Couple of concrete, terse, stable instructions for the model
    sys = SystemMessage(content=(
        "You are a deterministic command router for a Tableau assistant. "
        "Classify the user's intent and extract arguments into a strict schema. "
        "If filters are mentioned (e.g., Region=APAC 2024), normalize them into filters_json. "
        "If the user is asking you to remember/recall personal facts ('what is my...'), set intent=MEMORY_QA. "
        "If the user wants to PUBLISH a mock datasource, fill publish.project_name/datasource_name/overwrite when present. "
        "Prefer concrete values from the user text; do not hallucinate."
        "You are a deterministic command router for a Tableau assistant.\n"
        "- Map verbs:\n"
        "  * SEE/visualize verbs → intent=VIEW_IMAGE (e.g., 'show', 'display', 'render', 'image', 'screenshot', 'see', 'preview').\n"
        "  * DATA/analysis verbs → intent=VIEW_DATA (e.g., 'csv', 'data', 'table', 'rows', 'export', 'download', 'analyze', 'analysis').\n"
        "  * If both appear, DATA wins (csv/export beats show).\n"
        "- Extract arguments into the strict schema. If filters are mentioned (e.g., Region=APAC, Year=2024), normalize into filters_json.\n"
        "- If the user is asking to remember/recall personal facts ('what is my ...'), set intent=MEMORY_QA.\n"
        "- If the user wants to PUBLISH a mock datasource, fill publish.project_name/datasource_name/overwrite when present.\n"
        "- Prefer concrete values from the user text; do not hallucinate."
     ))

    # Keep the message short; provide a small slice of recalled context to help with references like 'same as before'
    snippet_block = ""
    if recalled_snippets:
        top = recalled_snippets[:6]
        snippet_block = "Relevant snippets:\n- " + "\n- ".join(top) + "\n"

    # Ask for structured output using Pydantic
    structured_llm = llm.with_structured_output(ParsedCommand)
    hm = HumanMessage(content=(
        f"{snippet_block}"
        "Return ONLY fields defined by the schema; avoid made-up keys.\n\n"
        f"User message:\n{user_text}"
    ))
    try:
        parsed: ParsedCommand = structured_llm.invoke([sys, hm])
        text_l = (user_text or "").lower()
        see_words = ("show", "display", "render", "image", "screenshot", "see", "preview")
        data_words = ("csv", "data", "table", "rows", "export", "download", "analyze", "analysis", "analytics")
        if parsed.intent in (Intent.UNKNOWN, Intent.VIEW_IMAGE, Intent.VIEW_DATA):
            if any(w in text_l for w in data_words):
                parsed.intent = Intent.VIEW_DATA
            elif any(w in text_l for w in see_words):
                parsed.intent = Intent.VIEW_IMAGE
        if isinstance(parsed.filters_json, str):
            try:
                fj = json.loads(parsed.filters_json)
                parsed.filters_json = fj if isinstance(fj, dict) else None
            except Exception:
                parsed.filters_json = None
        return parsed
    except Exception as e:
        log.warning("LLM parse failed, falling back to UNKNOWN: %s", e, exc_info=True)
        return ParsedCommand(intent=Intent.UNKNOWN, reason="LLM parse error")


# ---- Parse publish details from free text or JSON ----

SAVE_INTENT_RE = re.compile(r"\b(remember|save|note|store)\b", re.I)

def maybe_pin_fact(session_id: str, text: str):
    """
    If the user says 'remember/save/note/store ...', write a second, clearer
    schema-free memory line in LangMem to make retrieval easier, e.g.:
        [FACT] my favorite color is teal
    """
    if not text or not SAVE_INTENT_RE.search(text):
        return
    # Strip leading command words like "remember that", "save that", etc.
    m = re.match(r'^\s*(?:please\s+)?(?:remember|save|note|store)\s+(?:that\s+)?(.+)$', text, re.I)
    fact = (m.group(1) if m else text).strip()
    if fact:
        try:
            LANGMEM.add(session_id, f"[FACT] {fact}")
            log.debug("Pinned fact to LangMem: %s", _truncate(fact))
        except Exception as e:
            log.warning("Failed to pin fact: %s", e)



def ask_for_publish_names_hint() -> str:
    return (
        "Before I publish the mock datasource, what **project name** and **datasource name** should I use?"
        "\nYou can reply in either format:\n"
        "- `project=Team Analytics, datasource=Sandbox_Sales, overwrite=true`\n"
        "- `{\"project\": \"Team Analytics\", \"datasource\": \"Sandbox_Sales\", \"overwrite\": false}`"
    )

# ---------------- Prompt ----------------
SYSTEM_PROMPT = """You are a helpful Tableau assistant named Alex.

- If the user wants to SEE a chart, call `tableau_get_view_image`.
- If the user wants to ANALYZE a chart/table, call `tableau_get_view_data`, then answer from those rows.
- If the user wants to list all the views in a specific workbook (e.g., "List all views in Superstore"), call `list_tableau_views` with the workbook name.
- If the user says 'show', 'display', 'render', 'see', 'preview', or 'screenshot', you MUST use `tableau_get_view_image` (never the CSV endpoint) unless they also explicitly say 'csv', 'table', 'rows', 'export', or 'analyze'. In that case use `tableau_get_view_data`.
- Prefer using a tool over guessing. If filters are mentioned (e.g., Region=APAC, Year=2024), pass them as JSON to filters_json.
- Do NOT print raw data URLs or base64 in your final answer.
- When you fetch an image, just acknowledge it briefly; the UI will display it.

**Important (Publishing mock datasource):**
- Do NOT call `publish_mock_datasource` with default names.
- FIRST confirm `project_name` and `datasource_name`. If missing or default-like, ask the user:
  "What project name and datasource name should I use? (optional: overwrite=true/false)"
- Only after the user confirms the names, call the tool with those names.

**Memories (schema-free)**
- You will be shown a compact "User facts" block and raw "Recalled user memories".
- Treat both as authoritative context from the user. Use them to resolve references like
  "my/our/that/same as before" and previously stated facts/preferences/constraints.
- If a needed fact is present, do NOT ask again—use it directly. If it's missing, say so and invite the user to provide it.
- Answer directly from the recalled memories when the user asks about their own info (e.g., "what is my …", "do you remember …"). If the memory is insufficient, say so plainly instead of guessing.
"""

# ---------------- Memory layer ----------------
def _truncate(s: str, n: int = 200) -> str:
    return (s[: n - 3] + "...") if s and len(s) > n else (s or "")

# Tunables (override via env)
USER_TAIL_MAX = int(os.getenv("USER_TAIL_MAX", "600"))
RECALL_SEM_K = int(os.getenv("RECALL_SEM_K", "24"))
RECALL_TAIL_K = int(os.getenv("RECALL_TAIL_K", "24"))
RECALL_TOTAL_K = int(os.getenv("RECALL_TOTAL_K", "32"))
MEMORY_Q_TAIL_K = int(os.getenv("MEMORY_Q_TAIL_K", "64")) 

class UserTailMemory:
    """User-only recency tail per session (keeps just user utterances)."""
    def __init__(self, max_len: int = USER_TAIL_MAX):
        self._store: Dict[str, List[str]] = {}
        self._max_len = max_len

    def add_user(self, session_id: str, text: str):
        arr = self._store.setdefault(session_id, [])
        arr.append(text)
        if len(arr) > self._max_len:
            self._store[session_id] = arr[-self._max_len :]

    def last_n(self, session_id: str, n: int = RECALL_TAIL_K) -> List[str]:
        arr = self._store.get(session_id, [])
        res = arr[-n:]
        log.debug("UserTail.last_n session=%s n=%s -> %d", session_id, n, len(res))
        return res

class LangMemWrapper:
    """Semantic memory via LangMem + Gemini embeddings."""
    def __init__(self, model):
        self.available: bool = False
        self.manager = None
        self.store = None
        if create_memory_store_manager is None or InMemoryStore is None:
            log.warning("LangMem not initialized (missing manager or store).")
            return
        try:
            if GoogleGenerativeAIEmbeddings is None or not os.getenv("GOOGLE_API_KEY"):
                log.warning("Gemini embeddings unavailable; LangMem disabled.")
                return
            gem_model = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
            dims = int(os.getenv("EMBED_DIMS", "768"))
            embedder = GoogleGenerativeAIEmbeddings(
                model=gem_model, google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            def _to_list(x):
                if isinstance(x, str):
                    return [x]
                try:
                    return [str(t) for t in list(x)]
                except TypeError:
                    return [str(x)]

            def _embed(texts):
                items = _to_list(texts)
                return embedder.embed_documents(items)

            self.store = InMemoryStore(index={"dims": dims, "embed": _embed})
        except Exception as e:
            log.error("Failed to initialize LangMem: %s", e, exc_info=True)
            self.available = False

    def add(self, session_id: str, text: str) -> bool:
        if not self.available:
            log.debug("LangMem.add skipped (unavailable)."); return False
        try:
            cfg = {"configurable": {"langgraph_user_id": session_id}}
            log.debug("LangMem.add -> session=%s text=%s", session_id, _truncate(text))
            self.manager.invoke({"messages": [{"role": "user", "content": text}]}, config=cfg)
            log.info("LangMem.add stored OK: session=%s", session_id)
            return True
        except Exception as e:
            log.error("LangMem.add failed: session=%s err=%s", session_id, e, exc_info=True)
            return False

    def search(self, session_id: str, query: str, k: int = RECALL_SEM_K) -> List[str]:
        if not self.available:
            log.debug("LangMem.search skipped (unavailable)."); return []
        try:
            cfg = {"configurable": {"langgraph_user_id": session_id}}
            log.debug("LangMem.search -> session=%s k=%s query=%s", session_id, k, _truncate(query))
            results = self.manager.search(query=query, config=cfg, limit=k)
            out: List[str] = []
            for item in results or []:
                val = getattr(item, "value", None)
                if isinstance(val, dict):
                    text = val.get("text") or val.get("content") or json.dumps(val)
                elif val is not None:
                    text = str(val)
                else:
                    text = str(item)
                if text:
                    out.append(text)
            log.info("LangMem.search returned %d item(s) for session=%s", len(out), session_id)
            for i, t in enumerate(out):
                log.debug("LangMem.search[%d]: %s", i, _truncate(t, 300))
            return out[:k]
        except Exception as e:
            log.error("LangMem.search failed: session=%s err=%s", session_id, e, exc_info=True)
            return []

# Instantiate memory backends
USER_TAIL = UserTailMemory()
LANGMEM = LangMemWrapper(llm)

def memory_add(session_id: str, role: str, text: str):
    """Store every user utterance in LangMem + user-tail."""
    cleaned_text = (text or "").strip()
    if not cleaned_text: return
    if role == "user":
        stored = LANGMEM.add(session_id, cleaned_text)
        USER_TAIL.add_user(session_id, cleaned_text)
        log.debug("memory_add -> user stored: LangMem=%s, UserTail=YES", stored)
    else:
        log.debug("memory_add -> assistant text (not stored in user-tail).")

def memory_recall(session_id: str, query: str) -> List[str]:
    """Recall = LangMem semantic hits + user-tail (dedup, preserve order, cap)."""
    sem = LANGMEM.search(session_id, query, k=RECALL_SEM_K) or []
    tail = USER_TAIL.last_n(session_id, n=RECALL_TAIL_K) or []
    seen = set(); merged: List[str] = []
    for item in sem + tail:
        if item not in seen:
            merged.append(item); seen.add(item)
    out = merged[:RECALL_TOTAL_K]
    log.debug("memory_recall -> session=%s sem=%d tail=%d merged=%d (cap=%d)",
              session_id, len(sem), len(tail), len(out), RECALL_TOTAL_K)
    return out

# ---------------- Schema-free semantic memory QA ----------------
MEMORY_Q_RE = re.compile(
    r"\b("
    r"what\s+(?:is|was|are)\s+(?:my|our)\b|"
    r"what'?s\s+(?:my|our)\b|"
    r"do\s+you\s+remember\b|"
    r"what\s+did\s+i\s+say\b|"
    r"remind\s+me\s+what\b|"
    r"what\s+did\s+i\s+tell\s+you\b|"
    r"recall\s+(?:my|our)\b"
    r")",
    re.I,
)


def gather_memory_snippets(session_id: str, query: str, extra_k: int = 10) -> List[str]:
    """
    Search LangMem with several phrasings and return compact snippets.
    No schemas; just text. Include a bit of user-tail for recency.
    """
    qs = [
        query,
        f"User self info: {query}",
        f"My details: {query}",
        f"Earlier you said: {query}",
    ]
    out: List[str] = []
    seen = set()
    # semantic hits
    for q in qs:
        hits = LANGMEM.search(session_id, q, k=extra_k) or []
        for h in hits:
            if h and h not in seen:
                out.append(h); seen.add(h)
    # recency tail
    for t in USER_TAIL.last_n(session_id, n=MEMORY_Q_TAIL_K):
        if t and t not in seen:
            out.append(t); seen.add(t)
    return out[:24]

def _supported_by_snippets(ans: str, snippets: List[str]) -> bool:
    if not ans: return False
    al = ans.lower()
    # Require that at least one meaningful chunk of the answer appears in a snippet.
    # Split on punctuation/commas to find a short phrase to check.
    parts = [p.strip() for p in re.split(r"[.;,\n]", ans) if p.strip()]
    haystack = "\n".join(snippets).lower()
    # If any non-trivial part (>= 3 chars) is found verbatim in snippets, we accept.
    return any((len(p) >= 3 and p.lower() in haystack) for p in parts)

def try_semantic_memory_answer(session_id: str, user_text: str) -> Optional[str]:
    if not MEMORY_Q_RE.search(user_text or ""):
        return None

    snippets = gather_memory_snippets(session_id, user_text, extra_k=12)
    if not snippets:
        return "I couldn’t find that in our past messages. Tell me now and I’ll remember."

    guide = (
        "You are retrieving the user's previously stated information. "
        "Use ONLY the snippets below. If the answer isn't clearly present, reply exactly with 'INSUFFICIENT'.\n\n"
        "Snippets:\n- " + "\n- ".join(snippets) + "\n\nQuestion: " + user_text + "\nAnswer:"
    )
    resp: AIMessage = llm.invoke([SystemMessage(content="Be concise. Use only given snippets."),
                                  HumanMessage(content=guide)])
    ans = (resp.content or "").strip()
    if (not ans) or ans.upper().startswith("INSUFFICIENT") or not _supported_by_snippets(ans, snippets):
        return "I couldn’t find that in our past messages. Tell me now and I’ll remember."
    if re.match(r"\s*my\b", ans, re.IGNORECASE):
        return re.sub(r"^\s*my\b", "Your", ans, flags=re.IGNORECASE)
    return ans

# ---------------- Cleaning / attachments ----------------
JSON_CODEBLOCK_RE = re.compile(r"```(?:json|csv|table)?[\s\S]*?```", re.IGNORECASE)

def _clean_output_text(text: str, attachments) -> str:
    if not isinstance(text, str): return ""
    if text.strip().startswith("{") and text.strip().endswith("}"):
        try:
            _obj = json.loads(text)
            if isinstance(_obj, dict) and "text" in _obj and isinstance(_obj["text"], str):
                text = _obj["text"]
        except Exception:
            pass
    text = text.replace("Final Answer:", "").strip()
    text = re.sub(r'data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=\s]+', '', text).strip()
    text = JSON_CODEBLOCK_RE.sub("", text).strip()
    has_table = any(a.get("type") == "table" for a in (attachments or []))
    has_file = any(a.get("type") == "file" for a in (attachments or []))
    if has_table and not has_file:
        t = text.lstrip()
        if (t.startswith("[") or t.startswith("{")) or len(t) > 300:
            text = ""
    if attachments and not text:
        for a in attachments:
            if a.get("caption"): return a["caption"]
        if has_file:
            return "Here’s your CSV export."
        return "Here’s your data."
    return text

DATA_URL_RE = re.compile(r'(data:image/(?:png|jpeg|jpg|gif|webp);base64,[A-Za-z0-9+/=]+)')

def _write_csv(rows: list, columns: list, prefix: str = "tableau_export") -> str:
    """Write rows/columns to /mnt/data and return file path."""
    fname = f"/mnt/data/{prefix}_{uuid.uuid4().hex[:8]}.csv"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if columns:
            w.writerow(columns)
        for r in rows:
            if isinstance(r, dict):
                w.writerow([r.get(c, "") for c in columns])
            else:
                w.writerow(r)
    return fname

def _extract_attachments(tool_steps: List[Dict[str, Any]]):
    attachments = []
    for step in tool_steps:
        s = step.get("result")
        if not s: continue
        s = s if isinstance(s, str) else str(s)
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                img = obj.get("image")
                if isinstance(img, str) and img.startswith("data:image/"):
                    attachments.append({"type": "image", "dataUrl": img, "caption": obj.get("text", "")})
                tbl = obj.get("table")
                if isinstance(tbl, list) and tbl:
                    cols = obj.get("columns", [])
                    caption = obj.get("text", "")
                    attachments.append({
                        "type": "table",
                        "rows": tbl,
                        "columns": cols,
                        "caption": caption,
                    })
                    # Also create a CSV file attachment for download
                    csv_path = _write_csv(tbl, cols)
                    attachments.append({
                        "type": "file",
                        "mime": "text/csv",
                        "path": csv_path,
                        "filename": os.path.basename(csv_path),
                        "caption": caption or "Exported data",
                    })
                continue
        except Exception:
            pass
        m = DATA_URL_RE.search(s)
        if m:
            attachments.append({"type": "image", "dataUrl": m.group(1), "caption": ""})
    return attachments

# ---------------- Simple chat schema ----------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"  # scope memories/conversation

class ChatResponse(BaseModel):
    response: str
    attachments: Optional[List[Dict[str, Any]]] = None

# ---------------- Routes ----------------
@app.get("/")
def read_root():
    return {"status": "Tableau Chatbot Agent is running (LangMem + user-tail, schema-free memory)!"}

def _build_messages(req: ChatRequest, facts: List[str], recalled: List[str]):
    # We keep these blocks in place for compatibility; facts list is empty (no pinning),
    # but recalled snippets are injected for context.
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "(none)"
    memory_blob = "\n".join(f"- {m}" for m in recalled) if recalled else "(none)"
    log.debug("Build messages -> facts=%d, recalled %d item(s) for session=%s",
              len(facts), len(recalled), req.session_id or "default")
    sys = SystemMessage(content=SYSTEM_PROMPT +
                        f"\n\n[User facts]\n{facts_block}\n\n[Recalled user memories]\n{memory_blob}\n")
    user = HumanMessage(content=req.message)
    return [sys, user]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Free-form natural language with schema-free memory:
    0) Direct semantic-memory Q&A for "what is my ... / do you remember ...".
    1) Recall (LangMem semantic hits + user-tail).
    2) Tool call(s) if needed, then final natural response.
    3) Store the turn in memory for future questions.
    """
    session_id = request.session_id or "default"
    log.info("POST /chat session=%s msg=%s", session_id, _truncate(request.message, 300))

    # --- 0) Schema-free memory Q&A (LangMem-only) ---
    maybe_pin_fact(session_id, request.message or "")
    primary_recall = memory_recall(session_id, request.message)
    extra_recall = gather_memory_snippets(session_id, request.message, extra_k=6)
    # merged recalled
    seen = set(); recalled = []
    for s in primary_recall + extra_recall:
        if s not in seen:
            recalled.append(s); seen.add(s)

    parsed = _llm_parse_user_utterance(request.message or "", recalled)

    if parsed.intent in (Intent.VIEW_IMAGE, Intent.VIEW_DATA):
        tool_name = "tableau_get_view_image" if parsed.intent == Intent.VIEW_IMAGE else "tableau_get_view_data"
        args = {
            "view_name": parsed.view_name or "",
            "workbook_name": parsed.workbook_name or "",
            "filters_json": json.dumps(parsed.filters_json or {}),
        }
        if not args["view_name"]:
            pass
        else:
            try:
                result = TOOL_REGISTRY[tool_name].invoke(args)
            except Exception as e:
                result = f"Tool '{tool_name}' failed: {e}"
                log.error(result, exc_info=True)
            attachments = _extract_attachments([{"name": tool_name, "args": args, "result": result}])
            output_text = _clean_output_text(str(result), attachments)
            memory_add(session_id, "user", request.message)
            memory_add(session_id, "assistant", output_text)
            return {"response": output_text, "attachments": attachments}

    elif parsed.intent == Intent.PUBLISH_MOCK:
        pub = parsed.publish or PublishArgs()
        if pub.project_name and pub.datasource_name:
            args = {"project_name": pub.project_name, "datasource_name": pub.datasource_name}
            if pub.overwrite is not None: args["overwrite"] = bool(pub.overwrite)
            try:
                result = publish_mock_datasource.invoke(args)
            except Exception as e:
                result = f"Tool 'publish_mock_datasource' failed: {e}"
                log.error(result, exc_info=True)
            attachments = _extract_attachments([{"name": "publish_mock_datasource", "args": args, "result": result}])
            output_text = _clean_output_text(str(result), attachments)
            memory_add(session_id, "user", request.message)
            memory_add(session_id, "assistant", output_text)
            return {"response": output_text, "attachments": attachments}
        else:
            set_pending(session_id, "publish_mock_datasource")
            ask = ask_for_publish_names_hint()
            memory_add(session_id, "assistant", ask)
            return {"response": ask, "attachments": []}
    if parsed.intent == Intent.MEMORY_QA:
        mem_ans = try_semantic_memory_answer(session_id, request.message or "")
        if mem_ans is not None:
            memory_add(session_id, "user", request.message)
            memory_add(session_id, "assistant", mem_ans)
            return {"response": mem_ans, "attachments": []}

    # --- 1) recall (richer, schema-free) ---
    primary_recall = memory_recall(session_id, request.message)
    extra_recall = gather_memory_snippets(session_id, request.message, extra_k=6)
    seen = set(); recalled = []
    for s in primary_recall + extra_recall:
        if s not in seen:
            recalled.append(s); seen.add(s)
    facts_for_prompt: List[str] = []  # no pinning/schema; keep interface stable

    pending = get_pending(session_id)
    if pending and pending.get("intent") == "publish_mock_datasource":
        # Re-parse current message for publish slots
        parsed_now = _llm_parse_user_utterance(request.message or "", recalled)
        pub = (parsed_now.publish or PublishArgs())
        pn, dn, ow = pub.project_name, pub.datasource_name, pub.overwrite
        log.debug("Pending publish (LLM): pn=%s dn=%s ow=%s", pn, dn, ow)

        if pn and dn:
            args = {"project_name": pn, "datasource_name": dn}
            if ow is not None: args["overwrite"] = bool(ow)
            try:
                result = publish_mock_datasource.invoke(args)
            except Exception as e:
                result = f"Tool 'publish_mock_datasource' failed: {e}"
                log.error(result, exc_info=True)
            attachments = _extract_attachments([{"name": "publish_mock_datasource", "args": args, "result": result}])
            clear_pending(session_id)
            output_text = _clean_output_text(f"{result}", attachments)
            memory_add(session_id, "user", request.message)
            memory_add(session_id, "assistant", output_text)
            log.info("Publish flow complete.")
            return {"response": output_text, "attachments": attachments}
        else:
            ask = ask_for_publish_names_hint()
            memory_add(session_id, "user", request.message)
            memory_add(session_id, "assistant", ask)
            log.info("Publish flow: asking for names (LLM).")
            return {"response": ask, "attachments": []}


    # --- 2) LLM call: decide tool calls ---
    llm_with_tools = llm.bind_tools(TOOLS)
    msgs = _build_messages(request, facts_for_prompt, recalled)
    ai_first: AIMessage = llm_with_tools.invoke(msgs)
    log.debug("LLM first response length=%d, tool_calls=%s",
              len(ai_first.content or ""), bool(getattr(ai_first, "tool_calls", None)))

    # --- 3) execute tool calls (if any), then finalization call ---
    tool_steps: List[Dict[str, Any]] = []
    final_text = ai_first.content or ""
    if getattr(ai_first, "tool_calls", None):
        tool_msgs: List[ToolMessage] = []
        for tc in ai_first.tool_calls:
            name = tc["name"]; args = tc.get("args", {}) or {}
            tool = TOOL_REGISTRY.get(name)
            log.info("Tool call requested: %s args=%s", name, args)
            if not tool:
                log.warning("Unknown tool requested: %s", name); continue

            if name == "publish_mock_datasource":
                pub = parsed.publish or PublishArgs()
                pn = args.get("project_name") or pub.project_name
                dn = args.get("datasource_name") or pub.datasource_name
                ow = args.get("overwrite")
                if ow is None and pub.overwrite is not None:
                    ow = pub.overwrite

                # Guard against placeholder/defaults and missing names
                if not pn or not dn or pn == "AI Demos" or dn == "AI_Sample_Sales":
                    set_pending(session_id, "publish_mock_datasource")
                    ask = ask_for_publish_names_hint()
                    memory_add(session_id, "assistant", ask)
                    log.info("Blocked publish due to missing/placeholder names (LLM parse).")
                    return {"response": ask, "attachments": []}

                args = {"project_name": pn, "datasource_name": dn}
                if ow is not None:
                    args["overwrite"] = bool(ow)

            elif name in ("tableau_get_view_image", "tableau_get_view_data"):
                # Pre-fill slots for view tools
                if parsed.view_name and not args.get("view_name"):
                    args["view_name"] = parsed.view_name
                if parsed.workbook_name and not args.get("workbook_name"):
                    args["workbook_name"] = parsed.workbook_name
                if parsed.filters_json and not args.get("filters_json"):
                    try:
                        # ensure it's JSON string since your tool expects str
                        args["filters_json"] = json.dumps(parsed.filters_json)
                    except Exception:
                        pass

            try:
                result = tool.invoke(args)
                log.info("Tool executed: %s", name)
            except Exception as e:
                result = f"Tool '{name}' failed: {e}"
                log.error(result, exc_info=True)
            tool_steps.append({"name": name, "args": args, "result": result})
            tool_msgs.append(ToolMessage(tool_call_id=tc["id"], name=name, content=result))

        ai_final: AIMessage = llm_with_tools.invoke(msgs + [ai_first] + tool_msgs)
        final_text = ai_final.content or final_text
        log.debug("LLM final response length=%d", len(final_text or ""))

    # --- 4) attachments + output cleanup ---
    attachments = _extract_attachments(tool_steps)
    if not (final_text or "").strip() and tool_steps:
        # Use the tools' textual results when the model is silent
        for step in tool_steps:
            res = step.get("result")
            if not res:
                continue
            s = str(res)
            if s.strip().startswith("{") and s.strip().endswith("}"):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                        final_text = obj["text"]
                        break
                except Exception:
                    pass
            if s.strip():
                final_text = s
                break
    output_text = _clean_output_text(final_text, attachments)
    log.info("Final output length=%d (attachments=%d)", len(output_text or ""), len(attachments))

    # --- 5) store memories for future turns ---
    memory_add(session_id, "user", request.message)
    memory_add(session_id, "assistant", output_text)

    return {"response": output_text, "attachments": attachments}

# -------- Direct REST endpoints --------
@app.get("/views/image")
def get_view_image(view_name: str, workbook_name: str = "", filters_json: str = ""):
    payload = {"view_name": view_name, "workbook_name": workbook_name, "filters_json": filters_json}
    log.info("/views/image %s", payload)
    return json.loads(tableau_get_view_image.invoke(payload))

@app.get("/views/data")
def get_view_data(view_name: str, workbook_name: str = "", filters_json: str = "", max_rows: int = 200):
    payload = {"view_name": view_name, "workbook_name": workbook_name, "filters_json": filters_json, "max_rows": max_rows}
    log.info("/views/data %s", payload)
    return json.loads(tableau_get_view_data.invoke(payload))

@app.post("/datasources/publish-mock")
def api_publish_mock():
    log.info("/datasources/publish-mock (direct endpoint)")
    return {"message": publish_mock_datasource.invoke({})}
