
import json
import os
import re
from typing import List, Dict, Any, Optional,Tuple


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from tab_tools import (
    list_tableau_projects,
    list_tableau_workbooks,
    list_tableau_views,
    tableau_get_view_image,
    tableau_get_view_data,
    publish_mock_datasource
)

from fastapi import UploadFile, File, Form

# NEW: LangChain RAG pieces
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None

from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:
    GoogleGenerativeAIEmbeddings = None

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None

from langchain_community.vectorstores import Milvus

import tempfile


try:

    from mem0 import Mem0 
except Exception:
    Mem0 = None 

load_dotenv()

# ---------------- App setup ----------------
app = FastAPI(
    title="Tableau Chatbot Agent API",
    description="An API that uses a fast, single-shot tool caller to interact with Tableau for visualization + Q&A, with Mem0 memory.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://agent-tableau.onrender.com","http://localhost:5173","https://enmployee-os-gq96.vercel.app"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
RAG_COLLECTION_DEFAULT = os.getenv("RAG_COLLECTION", "tableau_docs")
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI")
ZILLIZ_CLOUD_TOKEN = os.getenv("ZILLIZ_CLOUD_API_KEY") or os.getenv("ZILLIZ_CLOUD_TOKEN")  # either name

# ---------------- LLM selection ----------------
def make_llm():
    """Pick an LLM based on available keys. Low temp for tool-use reliability."""
    # Try Google first
    if ChatGoogleGenerativeAI is not None:
        try:
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
        except Exception:
            pass
    # Fall back to OpenAI
    if ChatOpenAI is not None:
        try:
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception:
            pass
    raise RuntimeError(
        "No LLM configured. Set GOOGLE_API_KEY for Gemini or OPENAI_API_KEY for OpenAI."
    )

llm = make_llm()

def _require_rag_env():
    if not ZILLIZ_CLOUD_URI or not ZILLIZ_CLOUD_TOKEN:
        raise RuntimeError("ZILLIZ_CLOUD_URI / ZILLIZ_CLOUD_API_KEY missing in environment for RAG.")

def make_embedder():
    """
    Prefer Google embeddings (great quality, matches your Gemini usage).
    Fallback to OpenAI embeddings if available.
    """
    if GoogleGenerativeAIEmbeddings is not None and os.getenv("GOOGLE_API_KEY"):
        try:
            return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))
        except Exception:
            pass
    if OpenAIEmbeddings is not None and os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEmbeddings(model="text-embedding-3-large")
        except Exception:
            pass
    raise RuntimeError("No embeddings configured. Set GOOGLE_API_KEY (preferred) or OPENAI_API_KEY.")

EMBEDDINGS = None
VECTOR_STORES: Dict[str, Milvus] = {}  # cache by collection name
def _get_or_create_vs(collection_name: str) -> Milvus:
    """
    Return a connected Milvus vector store for collection_name.
    Creates the collection lazily on first upsert.
    """
    _require_rag_env()
    global EMBEDDINGS
    if EMBEDDINGS is None:
        EMBEDDINGS = make_embedder()

    if collection_name in VECTOR_STORES:
        return VECTOR_STORES[collection_name]

    # Connect to existing (or future) collection without inserting docs yet.
    # langchain_community Milvus supports constructing with just connection + collection.
    try:
        vs = Milvus(
            embedding_function=EMBEDDINGS,           # some versions accept 'embedding'; use both names below
            connection_args={"uri": ZILLIZ_CLOUD_URI, "token": ZILLIZ_CLOUD_TOKEN, "secure": True},
            collection_name=collection_name,
            auto_id=True,
        )
    except TypeError:
        # older signature
        vs = Milvus(
            embedding=EMBEDDINGS,
            connection_args={"uri": ZILLIZ_CLOUD_URI, "token": ZILLIZ_CLOUD_TOKEN, "secure": True},
            collection_name=collection_name,
        )
    VECTOR_STORES[collection_name] = vs
    return vs

def _split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def _load_pdf_bytes_to_docs(file_bytes: bytes, filename: str):
    if PyPDFLoader is None:
        raise RuntimeError("PyPDFLoader is not installed. pip install langchain-community pypdf")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".pdf") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        loader = PyPDFLoader(tmp.name)
        return loader.load()

def rag_upsert_pdf(file_bytes: bytes, filename: str, collection_name: str) -> int:
    """
    Ingest a single PDF: load -> split -> embed -> upsert to Milvus.
    Returns number of chunks inserted.
    """
    vs = _get_or_create_vs(collection_name)
    docs = _load_pdf_bytes_to_docs(file_bytes, filename)
    chunks = _split_docs(docs)
    # If the collection is new, from_documents is fastest; otherwise add_documents.
    if collection_name not in VECTOR_STORES or getattr(vs, "_is_empty", False):
        try:
            vs_new = Milvus.from_documents(
                documents=chunks,
                embedding=EMBEDDINGS if hasattr(EMBEDDINGS, "embed_query") else EMBEDDINGS,
                connection_args={"uri": ZILLIZ_CLOUD_URI, "token": ZILLIZ_CLOUD_TOKEN, "secure": True},
                collection_name=collection_name,
            )
            VECTOR_STORES[collection_name] = vs_new
        except Exception:
            # fallback to incremental add (works if collection already exists)
            vs.add_documents(chunks)
    else:
        vs.add_documents(chunks)
    return len(chunks)

def rag_search(question: str, collection_name: str, k: int = 4):
    vs = _get_or_create_vs(collection_name)
    return vs.similarity_search(question, k=k)

def _format_context(docs) -> str:
    ctx = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", meta.get("page_number", "?"))
        ctx.append(f"[{i}] (source={src}, page={page})\n{d.page_content}")
    return "\n\n".join(ctx)

def rag_answer_with_llm(question: str, collection_name: str, k: int = 4) -> Dict[str, Any]:
    """
    Retrieve, then ask your existing `llm` (ChatGoogleGenerativeAI or ChatOpenAI).
    """
    docs = rag_search(question, collection_name, k=k)
    context = _format_context(docs) if docs else "(no results)"
    sys = SystemMessage(content=(
        "You answer strictly from the provided context. "
        "If the answer is not present, say you don't know. "
        "Be concise, but accurate. Include short citations like [1], [2] when relevant."
    ))
    human = HumanMessage(content=f"Question:\n{question}\n\nContext:\n{context}")
    try:
        ai = llm.invoke([sys, human])
        text = (ai.content or "").strip()
    except Exception as e:
        text = f"RAG answering failed: {e}"

    # small inline citation map
    sources = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        sources.append({
            "rank": i,
            "source": meta.get("source"),
            "page": meta.get("page", meta.get("page_number")),
            "chars": len(d.page_content or ""),
        })
    return {"answer": text, "sources": sources, "k": k}
# ===================== RAG END (globals/helpers) =====================


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
    if s in ("true","yes","y","1"): return True
    if s in ("false","no","n","0"): return False
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
        if m: project_name = m.group(1).strip()
    if not datasource_name:
        m = _D_KV.search(text)
        if m: datasource_name = m.group(1).strip()
    if overwrite is None:
        m = _OW_KV.search(text)
        if m: overwrite = _to_bool(m.group(1))

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

# Instantiate Mem0 if available
MEM0_CLIENT = None
if Mem0:
    try:
        MEM0_CLIENT = Mem0(api_key=os.getenv("MEM0_API_KEY", ""))
    except Exception:
        MEM0_CLIENT = None

LOCAL_MEM = LocalMemory()

def memory_add(session_id: str, role: str, text: str):
    payload = f"[{role}] {text}"
    # Mem0 first
    if MEM0_CLIENT:
        try:
            # Different versions expose slightly different method names/args;
            # try common patterns safely:
            if hasattr(MEM0_CLIENT, "add"):
                MEM0_CLIENT.add(payload, user_id=session_id)
            elif hasattr(MEM0_CLIENT, "create"):
                MEM0_CLIENT.create(payload, user_id=session_id)
        except Exception:
            pass
    # Always mirror to local fallback
    LOCAL_MEM.add(session_id, payload)

def memory_search(session_id: str, query: str, k: int = 5) -> List[str]:
    results: List[str] = []
    # Mem0 retrieval
    if MEM0_CLIENT:
        try:
            if hasattr(MEM0_CLIENT, "search"):
                hits = MEM0_CLIENT.search(query, user_id=session_id, limit=k)
            elif hasattr(MEM0_CLIENT, "retrieve"):
                hits = MEM0_CLIENT.retrieve(query, user_id=session_id, k=k)
            else:
                hits = []
            for h in hits or []:
                # normalize shape differences
                txt = (
                    h.get("text")
                    or h.get("memory")
                    or h.get("content")
                    or str(h)
                )
                if txt:
                    results.append(txt)
        except Exception:
            pass
    # Fallback supplement from local
    if len(results) < k:
        results.extend(LOCAL_MEM.search(session_id, query, k=k - len(results)))
    return results[:k]

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

DATA_URL_RE = re.compile(
    r'(data:image/(?:png|jpeg|jpg|gif|webp);base64,[A-Za-z0-9+/=]+)'
)

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
                    attachments.append({
                        "type": "table",
                        "rows": tbl,
                        "columns": obj.get("columns", []),
                        "caption": obj.get("text", "")
                    })
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
    session_id: Optional[str] = "default"  # <-- NEW: scope memories/conversation

class ChatResponse(BaseModel):
    response: str
    attachments: Optional[List[Dict[str, Any]]] = None
class RAGQuery(BaseModel):
    question: str
    collection: Optional[str] = None
    k: int = 4

class RAGUploadResponse(BaseModel):
    ok: bool
    inserted: int
    collection: str

# ---------------- Routes ----------------
@app.get("/")
def read_root():
    return {"status": "Tableau Chatbot Agent is running (fast tool-caller + Mem0)!"}

def _build_messages(req: ChatRequest, recalled: List[str], rag_context: str = ""):
    memory_blob = "\n".join(f"- {m}" for m in recalled) if recalled else "(none)"
    kb_blob = ""
    if rag_context:
        kb_blob = (
            "\n\n[Knowledge Base Context]\n"
            f"{rag_context}\n\n"
            "When the user's question can be answered from the context above, "
            "answer directly from it and include short bracket citations like [1], [2] "
            "matching the numbered items in that context. If it is not relevant, proceed normally."
        )
    sys = SystemMessage(content=SYSTEM_PROMPT + f"\n\n[Recalled user memories]\n{memory_blob}\n" + kb_blob)
    user = HumanMessage(content=req.message)
    return [sys, user]

@app.post("/rag/upload", response_model=RAGUploadResponse)
async def rag_upload(
    file: UploadFile = File(...),
    collection: str = Form(None),
):
    """
    Upload a single PDF, chunk+embed, and upsert into Milvus.
    """
    _require_rag_env()
    coll = (collection or RAG_COLLECTION_DEFAULT).strip()
    data = await file.read()
    inserted = rag_upsert_pdf(data, file.filename, coll)
    return {"ok": True, "inserted": inserted, "collection": coll}

@app.post("/rag/query", response_model=ChatResponse)
async def rag_query(q: RAGQuery):
    """
    Ask a question against a collection; uses the same LLM for final answer.
    """
    coll = (q.collection or RAG_COLLECTION_DEFAULT).strip()
    result = rag_answer_with_llm(q.question, coll, k=q.k)
    # Store light memory of the turn (optional)
    memory_add("default", "user", f"[RAG] {q.question}")
    memory_add("default", "assistant", result["answer"])
    return ChatResponse(response=result["answer"], attachments=[{"type": "table", "rows": result["sources"], "columns": ["rank","source","page","chars"], "caption": "Top matches"}])

# ===== Add this helper somewhere above chat_endpoint =====
def _rag_context_and_sources(question: str, collection_name: str, k: int = 4):
    """
    Try to fetch relevant chunks for the question.
    Returns (context_str, sources_list). If RAG not configured or no hits, returns ("", []).
    """
    try:
        docs = rag_search(question, collection_name, k=k)
    except Exception:
        return "", []

    if not docs:
        return "", []

    context = _format_context(docs)
    sources = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        sources.append({
            "rank": i,
            "source": meta.get("source"),
            "page": meta.get("page", meta.get("page_number")),
            "chars": len(d.page_content or ""),
        })
    return context, sources


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    msg = (request.message or "").strip()

    # ---- 1) "doc:" short-circuit remains ----
    if msg.lower().startswith("doc:"):
        q = msg[4:].strip()
        coll = RAG_COLLECTION_DEFAULT
        result = rag_answer_with_llm(q, coll, k=4)
        memory_add(request.session_id or "default", "user", f"[RAG] {q}")
        memory_add(request.session_id or "default", "assistant", result["answer"])
        return ChatResponse(
            response=result["answer"],
            attachments=[{"type": "table", "rows": result["sources"], "columns": ["rank","source","page","chars"], "caption": "Top matches"}]
        )

    # ---- 2) recall memory + pending publish flow (unchanged) ----
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
            attachments = _extract_attachments([{"name": "publish_mock_datasource", "args": args, "result": result}])
            clear_pending(session_id)
            output_text = _clean_output_text(f"{result}", attachments)
            memory_add(session_id, "user", request.message)
            memory_add(session_id, "assistant", output_text)
            return {"response": output_text, "attachments": attachments}
        else:
            ask = ask_for_publish_names_hint()
            memory_add(session_id, "assistant", ask)
            return {"response": ask, "attachments": []}

    # ---- 3) NEW: try to pull RAG context automatically ----
    rag_context, rag_sources = _rag_context_and_sources(msg, RAG_COLLECTION_DEFAULT, k=4)

    # ---- 4) build messages (now include RAG context when present) ----
    llm_with_tools = llm.bind_tools(TOOLS)
    msgs = _build_messages(request, recalled, rag_context=rag_context)

    ai_first: AIMessage = llm_with_tools.invoke(msgs)

    tool_steps: List[Dict[str, Any]] = []
    final_text = ai_first.content or ""

    # ---- 5) tool calling loop (unchanged) ----
    if getattr(ai_first, "tool_calls", None):
        tool_msgs: List[ToolMessage] = []
        for tc in ai_first.tool_calls:
            name = tc["name"]
            args = tc.get("args", {}) or {}
            tool = TOOL_REGISTRY.get(name)
            if not tool:
                continue

            if name == "publish_mock_datasource":
                user_pn, user_dn, user_ow = parse_publish_details(request.message)
                pn = args.get("project_name") or user_pn
                dn = args.get("datasource_name") or user_dn
                ow = args.get("overwrite")
                if ow is None and user_ow is not None:
                    ow = user_ow

                if not pn or not dn or pn == "AI Demos" or dn == "AI_Sample_Sales":
                    set_pending(session_id, "publish_mock_datasource")
                    ask = ask_for_publish_names_hint()
                    memory_add(session_id, "assistant", ask)
                    return {"response": ask, "attachments": []}

                args = {"project_name": pn, "datasource_name": dn}
                if ow is not None:
                    args["overwrite"] = bool(ow)

            try:
                result = tool.invoke(args)
            except Exception as e:
                result = f"Tool '{name}' failed: {e}"
            tool_steps.append({"name": name, "args": args, "result": result})
            tool_msgs.append(ToolMessage(tool_call_id=tc["id"], name=name, content=result))

        ai_final: AIMessage = llm_with_tools.invoke(msgs + [ai_first] + tool_msgs)
        final_text = ai_final.content or final_text

    # ---- 6) Build attachments: tool outputs + (NEW) RAG sources ----
    attachments = _extract_attachments(tool_steps)
    if rag_sources:
        attachments.append({
            "type": "table",
            "rows": rag_sources,
            "columns": ["rank","source","page","chars"],
            "caption": "Top matches (RAG)"
        })

    output_text = _clean_output_text(final_text, attachments)

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
    payload = {"view_name": view_name, "workbook_name": workbook_name, "filters_json": filters_json, "max_rows": max_rows}
    return json.loads(tableau_get_view_data.invoke(payload))

@app.post("/datasources/publish-mock")
def api_publish_mock():
    return {"message": publish_mock_datasource.invoke({})}

