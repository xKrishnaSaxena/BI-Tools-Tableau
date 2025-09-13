"""
main.py — Tableau Chatbot Agent (LangMem + user-tail) — schema-free memory
- No predefined fields; remembers any user-stated info.
- Recall = LangMem semantic hits + last-N user utterances (dedup).
- CSV export for any table result (file attachment).
"""

import json
import os
import re
import csv
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# ---- Parse publish details from free text or JSON ----
_P_KV = re.compile(r'(?:\bproject(?:_name)?\b)\s*[:=]\s*["\']?([^,\n;]+?)["\']?(?=,|;|\s|$)', re.I)
_D_KV = re.compile(r'(?:\bdatasource(?:_name)?\b|data\s*source)\s*[:=]\s*["\']?([^,\n;]+?)["\']?(?=,|;|\s|$)', re.I)
_OW_KV = re.compile(r'\boverwrite\b\s*[:=]\s*(true|false|yes|no|y|n|1|0)', re.I)

def _to_bool(val: Any) -> Optional[bool]:
    if isinstance(val, bool): return val
    s = str(val).strip().lower()
    if s in ("true", "yes", "y", "1"): return True
    if s in ("false", "no", "n", "0"): return False
    return None

def parse_publish_details(text: str) -> Tuple[Optional[str], Optional[str], Optional[bool]]:
    project_name = None; datasource_name = None; overwrite = None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            project_name = obj.get("project") or obj.get("project_name") or project_name
            datasource_name = obj.get("datasource") or obj.get("datasource_name") or datasource_name
            if "overwrite" in obj: overwrite = _to_bool(obj.get("overwrite"))
    except Exception:
        pass
    if not project_name:
        m = _P_KV.search(text);  project_name = m.group(1).strip() if m else project_name
    if not datasource_name:
        m = _D_KV.search(text);  datasource_name = m.group(1).strip() if m else datasource_name
    if overwrite is None:
        m = _OW_KV.search(text); overwrite = _to_bool(m.group(1)) if m else None
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
RECALL_SEM_K = int(os.getenv("RECALL_SEM_K", "12"))
RECALL_TAIL_K = int(os.getenv("RECALL_TAIL_K", "12"))
RECALL_TOTAL_K = int(os.getenv("RECALL_TOTAL_K", "24"))

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
            def _embed(texts): return embedder.embed_documents(list(texts))
            self.store = InMemoryStore(index={"dims": dims, "embed": _embed})
            self.manager = create_memory_store_manager(
                model, namespace=("memories", "{langgraph_user_id}"), store=self.store, query_limit=5
            )
            self.available = True
            log.info("LangMem initialized: model=%s dims=%s", gem_model, dims)
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
    r"\b(what\s+is\s+my\b|do\s+you\s+remember\b|what\s+did\s+i\s+say\b|remind\s+me\s+what\b)",
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
    for t in USER_TAIL.last_n(session_id, n=8):
        if t and t not in seen:
            out.append(t); seen.add(t)
    return out[:24]

def try_semantic_memory_answer(session_id: str, user_text: str) -> Optional[str]:
    """
    If the user is asking about their own info, answer purely from LangMem snippets.
    Returns a short answer or None to fall back to normal flow.
    """
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
    if not ans or ans.upper().startswith("INSUFFICIENT"):
        return "I couldn’t find that in our past messages. Tell me now and I’ll remember."
    return ans

# ---------------- Cleaning / attachments ----------------
JSON_CODEBLOCK_RE = re.compile(r"```(?:json|csv|table)?[\s\S]*?```", re.IGNORECASE)

def _clean_output_text(text: str, attachments) -> str:
    if not isinstance(text, str): return ""
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
        pn, dn, ow = parse_publish_details(request.message)
        log.debug("Pending publish: parsed pn=%s dn=%s ow=%s", pn, dn, ow)
        if pn and dn:
            args = {"project_name": pn, "datasource_name": dn}
            if ow is not None: args["overwrite"] = ow
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
            log.info("Publish flow complete: %s", _truncate(output_text))
            return {"response": output_text, "attachments": attachments}
        else:
            ask = ask_for_publish_names_hint()
            memory_add(session_id, "assistant", ask)
            log.info("Publish flow: asking for names.")
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
                user_pn, user_dn, user_ow = parse_publish_details(request.message)
                pn = args.get("project_name") or user_pn
                dn = args.get("datasource_name") or user_dn
                ow = args.get("overwrite")
                if ow is None and user_ow is not None: ow = user_ow
                if not pn or not dn or pn == "AI Demos" or dn == "AI_Sample_Sales":
                    set_pending(session_id, "publish_mock_datasource")
                    ask = ask_for_publish_names_hint()
                    memory_add(session_id, "assistant", ask)
                    log.info("Blocked publish due to missing/placeholder names.")
                    return {"response": ask, "attachments": []}
                args = {"project_name": pn, "datasource_name": dn}
                if ow is not None: args["overwrite"] = bool(ow)

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
