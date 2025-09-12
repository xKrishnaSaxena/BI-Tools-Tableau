"""
main.py — Tableau Chatbot Agent (LangMem + Gemini embeddings)
- Mem0 removed
- LangMem for semantic memory (add/search)
- Gemini (Google Generative AI) used for both chat and embeddings
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

# ---------------- Optional LLM backends ----------------
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except Exception:
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None

try:
    from langchain_openai import ChatOpenAI  # fallback only if Gemini unavailable
except Exception:
    ChatOpenAI = None

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# ---------------- Tableau tool shims (imported as before) ----------------
from tab_tools import (
    list_tableau_projects,
    list_tableau_workbooks,
    list_tableau_views,
    tableau_get_view_image,
    tableau_get_view_data,
    publish_mock_datasource,
)

# ---------------- LangMem (new) ----------------
# Lightweight replacement for Mem0 with add/search semantics
try:
    from langmem import create_memory_store_manager  # type: ignore
    from langgraph.store.memory import InMemoryStore  # type: ignore
except Exception:
    create_memory_store_manager = None  # type: ignore
    InMemoryStore = None  # type: ignore

load_dotenv()

# ---------------- App setup ----------------
app = FastAPI(
    title="Tableau Chatbot Agent API",
    description=(
        "An API that uses a fast, single-shot tool caller to interact with Tableau for visualization + Q&A, "
        "with LangMem long-term memory (Gemini embeddings)."
    ),
    version="2.2.0",
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
    """Pick an LLM based on available keys. Low temp for tool-use reliability."""
    # Prefer Google Gemini
    if ChatGoogleGenerativeAI is not None and os.getenv("GOOGLE_API_KEY"):
        try:
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
        except Exception:
            pass
    # Fall back to OpenAI only if Gemini isn't configured
    if ChatOpenAI is not None and os.getenv("OPENAI_API_KEY"):
        try:
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception:
            pass
    raise RuntimeError(
        "No LLM configured. Set GOOGLE_API_KEY for Gemini (preferred) or OPENAI_API_KEY for OpenAI."
    )

llm = make_llm()

# ---------------- Tools wiring (same tools, no agent) ----------------
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
    PENDING[session_id] = {"intent": intent, "data": data or {}}

def clear_pending(session_id: str) -> None:
    if session_id in PENDING:
        del PENDING[session_id]

# ---- Parse publish details from free text or JSON ----
_P_KV = re.compile(r'(?:\bproject(?:_name)?\b)\s*[:=]\s*["\']?([^,\n;]+?)["\']?(?=,|;|\s|$)', re.I)
_D_KV = re.compile(r'(?:\bdatasource(?:_name)?\b|data\s*source)\s*[:=]\s*["\']?([^,\n;]+?)["\']?(?=,|;|\s|$)', re.I)
_OW_KV = re.compile(r'\boverwrite\b\s*[:=]\s*(true|false|yes|no|y|n|1|0)', re.I)

def _to_bool(val: Any) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "yes", "y", "1"):
        return True
    if s in ("false", "no", "n", "0"):
        return False
    return None

def parse_publish_details(text: str) -> Tuple[Optional[str], Optional[str], Optional[bool]]:
    project_name = None
    datasource_name = None
    overwrite = None

    # Try JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            project_name = obj.get("project") or obj.get("project_name") or project_name
            datasource_name = obj.get("datasource") or obj.get("datasource_name") or datasource_name
            if "overwrite" in obj:
                overwrite = _to_bool(obj.get("overwrite"))
    except Exception:
        pass

    # Key=value fallbacks
    if not project_name:
        m = _P_KV.search(text)
        if m:
            project_name = m.group(1).strip()
    if not datasource_name:
        m = _D_KV.search(text)
        if m:
            datasource_name = m.group(1).strip()
    if overwrite is None:
        m = _OW_KV.search(text)
        if m:
            overwrite = _to_bool(m.group(1))

    return project_name, datasource_name, overwrite


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
- Prefer using a tool over guessing. If filters are mentioned (e.g., Region=APAC, Year=2024), pass them as JSON to filters_json.
- Do NOT print raw data URLs or base64 in your final answer.
- When you fetch an image, just acknowledge it briefly; the UI will display it.

**Important (Publishing mock datasource):**
- Do NOT call `publish_mock_datasource` with default names.
- FIRST confirm `project_name` and `datasource_name`. If missing or default-like, ask the user:
  "What project name and datasource name should I use? (optional: overwrite=true/false)"
- Only after the user confirms the names, call the tool with those names.

You have access to long-term user memories (if provided below). Use them briefly and relevantly.
"""

# ---------------- Memory layer ----------------
class LocalMemory:
    """Simple in-process memory fallback keyed by session_id."""

    def __init__(self, k: int = 50):
        self._store: Dict[str, List[str]] = {}
        self._k = k

    def add(self, session_id: str, text: str):
        arr = self._store.setdefault(session_id, [])
        arr.append(text)
        if len(arr) > 500:  # prevent unbounded growth
            del arr[: len(arr) - 500]

    def search(self, session_id: str, query: str, k: int = 5) -> List[str]:
        # naive relevance: return last k snippets
        arr = self._store.get(session_id, [])
        return arr[-k:]


class LangMemWrapper:
    """Encapsulates LangMem manager+store to provide add/search like Mem0.

    - Uses LangGraph's InMemoryStore for persistence in-process.
    - Vector search enabled via **Gemini embeddings** (text-embedding-004).
    - If anything fails, `.available` stays False and we silently fall back to LocalMemory.
    """

    def __init__(self, model):
        self.available: bool = False
        self.manager = None
        self.store = None
        self._embed_f = None
        if create_memory_store_manager is None or InMemoryStore is None:
            return
        try:
            if GoogleGenerativeAIEmbeddings is None or not os.getenv("GOOGLE_API_KEY"):
                # Gemini embeddings not available; skip LangMem and rely on LocalMemory
                self.available = False
                return

            # --- Gemini embeddings ---
            gem_model = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
            dims = int(os.getenv("EMBED_DIMS", "768"))  # text-embedding-004 = 768 dims

            embedder = GoogleGenerativeAIEmbeddings(
                model=gem_model, google_api_key=os.getenv("GOOGLE_API_KEY")
            )

            def _embed(texts):
                # InMemoryStore expects a callable: Iterable[str] -> List[List[float]]
                return embedder.embed_documents(list(texts))

            self._embed_f = _embed
            self.store = InMemoryStore(index={"dims": dims, "embed": _embed})

            # Accept a BaseChatModel instance directly
            self.manager = create_memory_store_manager(
                model,
                namespace=("memories", "{langgraph_user_id}"),
                store=self.store,
                query_limit=5,
            )
            self.available = True
        except Exception:
            # If initialization fails (missing deps/keys), disable LangMem usage
            self.available = False

    def add(self, session_id: str, role: str, text: str) -> bool:
        if not self.available or role != "user":
            return False
        try:
            cfg = {"configurable": {"langgraph_user_id": session_id}}
            # Provide a minimal single-turn conversation; manager extracts memories and persists to store
            self.manager.invoke({"messages": [{"role": "user", "content": text}]}, config=cfg)
            return True
        except Exception:
            return False

    def search(self, session_id: str, query: str, k: int = 5) -> List[str]:
        if not self.available:
            return []
        try:
            cfg = {"configurable": {"langgraph_user_id": session_id}}
            results = self.manager.search(query=query, config=cfg, limit=k)
            out: List[str] = []
            for item in results or []:
                # Items may be strings, dicts, or SearchItem objects
                val = getattr(item, "value", None)
                if isinstance(val, dict):
                    text = val.get("text") or val.get("content") or json.dumps(val)
                elif val is not None:
                    text = str(val)
                else:
                    text = str(item)
                if text:
                    out.append(text)
            return out[:k]
        except Exception:
            return []


# Instantiate memory backends
LOCAL_MEM = LocalMemory()
LANGMEM = LangMemWrapper(llm)


def memory_add(session_id: str, role: str, text: str):
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return
    # Try LangMem first (store only user messages), else fall back to local
    used_langmem = LANGMEM.add(session_id, role, cleaned_text)
    # Always keep a short local tail for robustness
    if role == "user":
        LOCAL_MEM.add(session_id, f"[user] {cleaned_text}")
    else:
        LOCAL_MEM.add(session_id, f"[assistant] {cleaned_text}")


def memory_search(session_id: str, query: str, k: int = 5) -> List[str]:
    # Prefer LangMem semantic search; if unavailable or empty, supplement from LocalMemory
    primary = LANGMEM.search(session_id, query, k=k)
    if len(primary) < k:
        tail = LOCAL_MEM.search(session_id, query, k=k - len(primary))
        return (primary or []) + tail
    return primary[:k]

# ---------------- Cleaning / attachments (unchanged) ----------------
JSON_CODEBLOCK_RE = re.compile(r"```(?:json|csv|table)?[\s\S]*?```", re.IGNORECASE)


def _clean_output_text(text: str, attachments) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("Final Answer:", "").strip()
    text = re.sub(r'data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=\s]+', '', text).strip()
    text = JSON_CODEBLOCK_RE.sub("", text).strip()

    has_table = any(a.get("type") == "table" for a in (attachments or []))
    if has_table:
        t = text.lstrip()
        if (t.startswith("[") or t.startswith("{")) or len(t) > 300:
            text = ""

    if attachments and not text:
        for a in attachments:
            if a.get("caption"):
                return a["caption"]
        return "Here’s your data."
    return text


DATA_URL_RE = re.compile(r'(data:image/(?:png|jpeg|jpg|gif|webp);base64,[A-Za-z0-9+/=]+)')


def _extract_attachments(tool_steps: List[Dict[str, Any]]):
    """
    tool_steps: list of dicts with {name, args, result}
    Extract images/tables from tool results.
    """
    attachments = []
    for step in tool_steps:
        s = step.get("result")
        if not s:
            continue
        s = s if isinstance(s, str) else str(s)
        # Try strict JSON payloads first
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                img = obj.get("image")
                if isinstance(img, str) and img.startswith("data:image/"):
                    attachments.append({"type": "image", "dataUrl": img, "caption": obj.get("text", "")})
                tbl = obj.get("table")
                if isinstance(tbl, list) and tbl:
                    attachments.append(
                        {
                            "type": "table",
                            "rows": tbl,
                            "columns": obj.get("columns", []),
                            "caption": obj.get("text", ""),
                        }
                    )
                continue
        except Exception:
            pass
        # Fallback: scan for data URLs in free text
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
    return {"status": "Tableau Chatbot Agent is running (fast tool-caller + LangMem/Gemini)!"}


def _build_messages(req: ChatRequest, recalled: List[str]):
    memory_blob = "\n".join(f"- {m}" for m in recalled) if recalled else "(none)"
    sys = SystemMessage(content=SYSTEM_PROMPT + f"\n\n[Recalled user memories]\n{memory_blob}\n")
    user = HumanMessage(content=req.message)
    return [sys, user]


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Free-form natural language. Single-turn tool calling for speed.
    1) Recall memory (LangMem/local) relevant to the query.
    2) Let the LLM decide whether to call ONE OR MORE tools.
    3) Execute tools, provide results, and get a final natural response.
    4) Store the turn in memory for future questions.
    """
    # --- 1) recall memory relevant to the current query ---
    recalled = memory_search(request.session_id or "default", request.message, k=5)
    session_id = request.session_id or "default"
    pending = get_pending(session_id)

    if pending and pending.get("intent") == "publish_mock_datasource":
        pn, dn, ow = parse_publish_details(request.message)
        if pn and dn:
            args = {"project_name": pn, "datasource_name": dn}
            if ow is not None:
                args["overwrite"] = ow
            result = publish_mock_datasource.invoke(args)
            attachments = _extract_attachments(
                [{"name": "publish_mock_datasource", "args": args, "result": result}]
            )
            clear_pending(session_id)
            output_text = _clean_output_text(f"{result}", attachments)
            memory_add(session_id, "user", request.message)
            memory_add(session_id, "assistant", output_text)
            return {"response": output_text, "attachments": attachments}
        else:
            # Ask again with explicit format help
            ask = ask_for_publish_names_hint()
            memory_add(session_id, "assistant", ask)
            return {"response": ask, "attachments": []}

    # --- 2) first LLM call: decide tool calls ---
    llm_with_tools = llm.bind_tools(TOOLS)
    msgs = _build_messages(request, recalled)
    ai_first: AIMessage = llm_with_tools.invoke(msgs)

    # --- 3) execute tool calls (if any), then finalization call ---
    tool_steps: List[Dict[str, Any]] = []
    final_text = ai_first.content or ""
    if getattr(ai_first, "tool_calls", None):
        tool_msgs: List[ToolMessage] = []
        for tc in ai_first.tool_calls:
            name = tc["name"]
            args = tc.get("args", {}) or {}
            tool = TOOL_REGISTRY.get(name)
            if not tool:
                continue

            # Enforce asking for names before publishing
            if name == "publish_mock_datasource":
                user_pn, user_dn, user_ow = parse_publish_details(request.message)
                pn = args.get("project_name") or user_pn
                dn = args.get("datasource_name") or user_dn
                ow = args.get("overwrite")
                if ow is None and user_ow is not None:
                    ow = user_ow

                # If missing OR default-like, ask first
                if not pn or not dn or pn == "AI Demos" or dn == "AI_Sample_Sales":
                    set_pending(session_id, "publish_mock_datasource")
                    ask = ask_for_publish_names_hint()
                    memory_add(session_id, "assistant", ask)
                    return {"response": ask, "attachments": []}

                # Otherwise, force-clean args to confirmed names
                args = {"project_name": pn, "datasource_name": dn}
                if ow is not None:
                    args["overwrite"] = bool(ow)

            # Normal tool execution
            try:
                result = tool.invoke(args)  # returns str (often JSON)
            except Exception as e:
                result = f"Tool '{name}' failed: {e}"
            tool_steps.append({"name": name, "args": args, "result": result})
            tool_msgs.append(ToolMessage(tool_call_id=tc["id"], name=name, content=result))
        # second LLM call to produce the final answer
        ai_final: AIMessage = llm_with_tools.invoke(msgs + [ai_first] + tool_msgs)
        final_text = ai_final.content or final_text

    # --- 4) attachments + output cleanup ---
    attachments = _extract_attachments(tool_steps)
    output_text = _clean_output_text(final_text, attachments)

    # --- 5) store memories for future turns ---
    memory_add(request.session_id or "default", "user", request.message)
    memory_add(request.session_id or "default", "assistant", output_text)

    return {"response": output_text, "attachments": attachments}


# -------- Direct REST endpoints (unchanged) --------
@app.get("/views/image")
def get_view_image(view_name: str, workbook_name: str = "", filters_json: str = ""):
    payload = {"view_name": view_name, "workbook_name": workbook_name, "filters_json": filters_json}
    return json.loads(tableau_get_view_image.invoke(payload))


@app.get("/views/data")
def get_view_data(view_name: str, workbook_name: str = "", filters_json: str = "", max_rows: int = 200):
    payload = {
        "view_name": view_name,
        "workbook_name": workbook_name,
        "filters_json": filters_json,
        "max_rows": max_rows,
    }
    return json.loads(tableau_get_view_data.invoke(payload))


@app.post("/datasources/publish-mock")
def api_publish_mock():
    return {"message": publish_mock_datasource.invoke({})}
