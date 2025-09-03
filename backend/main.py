import json
import re
import os
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LLMs
from dotenv import load_dotenv

# Prefer Google; fall back to OpenAI if configured
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool

# Mem0
from mem0 import MemoryClient

from tab_tools import (
    list_tableau_projects,
    list_tableau_workbooks,
    list_tableau_views,
    tableau_get_view_image,
    tableau_get_view_data,
    publish_mock_datasource,
    refresh_datasource_now,
    create_hourly_refresh_schedule,
    tableau_job_status,
)

print("BOOT: loading .env ...")
load_dotenv()

MEM0_API = os.getenv("MEM0_API_KEY")
USER_ID = os.getenv("USER_ID", "default_user")
print("BOOT: MEM0_API_KEY loaded?", bool(MEM0_API))
print("BOOT: USER_ID =", USER_ID)

if not MEM0_API:
    raise SystemExit("Missing MEM0_API_KEY in .env")

# Initialize Mem0 client
memory_client = MemoryClient(api_key=MEM0_API)

# Mem0 Helper Functions
def normalize_text_from_mem(m: Any) -> str:
    if isinstance(m, dict):
        return (m.get("memory") or m.get("text") or "").strip()
    return str(m).strip()

def pick_relevant_memory(query: str, mems: List[Dict[str, Any]], min_score: float = 0.55) -> Optional[str]:
    q = query.lower()
    q_tokens = {tok for tok in re.findall(r"[a-zA-Z]{3,}", q)}
    candidates = []
    for m in mems[:5]:
        t = normalize_text_from_mem(m)
        if not t:
            continue
        t_low = t.lower()
        score = (m.get("score") if isinstance(m, dict) else None) or 0.0
        overlap = len(q_tokens.intersection(set(re.findall(r"[a-zA-Z]{3,}", t_low))))
        candidates.append((score, overlap, t))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for score, overlap, t in candidates:
        if score >= min_score or overlap >= 1:
            return t
    return None

def extract_value_from_memory(query: str, memory_text: str) -> Optional[str]:
    q = query.lower().strip()
    t = (memory_text or "").strip()
    if any(k in q for k in ["my name", "what is my name", "whats my name", "what's my name"]):
        m = re.search(r"(?:user\s+name\s+is|my\s+name\s+is)\s+([A-Za-z][A-Za-z\s'-]+)", t, re.I)
        if m:
            return m.group(1).strip()
    if any(k in q for k in ["my age", "what is my age", "how old am i"]):
        m = re.search(r"(?:user\s+age\s+is|my\s+age\s+is|i\s*am|i'm)\s*(\d{1,3})", t, re.I)
        if m:
            return m.group(1).strip()
    if any(k in q for k in ["where do i live", "my city", "my location", "where am i from"]):
        m = re.search(r"(?:i\s+live\s+in|my\s+city\s+is|user\s+city\s+is)\s+([A-Za-z][A-Za-z\s,'-]+)", t, re.I)
        if m:
            return m.group(1).strip()
    m = re.search(r"\bis\b\s+(.+)$", t, re.I)
    if m:
        val = m.group(1).strip()
        val = re.sub(r"[.\s]+$", "", val)
        return val if val else None
    return None

PERSONAL_ATTR_ALIASES = {
    "name": ["name", "fullname"],
    "age": ["age"],
    "city": ["city", "location", "hometown"],
    "country": ["country", "nation"],
    "email": ["email", "mail"],
    "phone": ["phone", "mobile", "number"],
    "order": ["order", "order number", "order id", "ticket", "case"],
    "company": ["company", "employer", "org", "organization"],
    "role": ["role", "title", "position", "designation"],
    "preference": ["preference", "favorite", "favourite", "likes", "dislikes"],
}

def best_attr_key(attr_word: str) -> Optional[str]:
    aw = attr_word.lower()
    for key, aliases in PERSONAL_ATTR_ALIASES.items():
        if aw == key or aw in aliases:
            return key
    return None

def get_attr_from_mem0(recall_fn, attr_key: str) -> Optional[str]:
    queries = [attr_key, f"my {attr_key}", f"user {attr_key}"]
    for q in queries:
        mems = recall_fn(q, k=5) or []
        for m in mems:
            t = normalize_text_from_mem(m)
            val = extract_value_from_memory(attr_key, t)
            if val:
                return val
        for m in mems:
            t = normalize_text_from_mem(m)
            if t:
                return t
    return None

def resolve_personal_slots(query: str, recall_fn) -> Tuple[str, Dict[str, str]]:
    resolved: Dict[str, str] = {}
    mentions = re.findall(r"\bmy\s+([a-zA-Z][a-zA-Z0-9_-]{1,30})\b", query, re.I)
    seen = set()
    attrs = [m for m in mentions if not (m.lower() in seen or seen.add(m.lower()))]
    for raw_attr in attrs:
        key = best_attr_key(raw_attr) or raw_attr.lower()
        val = get_attr_from_mem0(recall_fn, key)
        if val:
            resolved[key] = val
    if not resolved:
        return (query, resolved)
    ctx_parts = [f"{k}={v}" for k, v in resolved.items()]
    context = " | ".join(ctx_parts)
    augmented = (
        f"Question: {query}\n"
        f"User context: {context}\n"
        f"Answer the question succinctly using the context when helpful."
    )
    return augmented, resolved

# Mem0 Adapter
class Mem0ManagedAdapter:
    def __init__(self, api_key: str):
        self.client = MemoryClient(api_key=api_key)
        print("BOOT: dir(mem0.client) ->", [m for m in dir(self.client) if not m.startswith("_")])
        print("BOOT: using client.add/search")

    def save(self, user_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        try:
            messages = [{"role": "user", "content": text}]
            self.client.add(messages=messages, user_id=user_id, version="v2", output_format="v1.1")
        except Exception as e:
            print(f"WARN: Failed to save memory: {e}")

    def search(self, user_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            filters = {"user_id": user_id}  # Required for v2 API
            hits = self.client.search(query=query, filters=filters, version="v2", limit=k)
            if isinstance(hits, dict) and "data" in hits:
                hits = hits["data"]
            if isinstance(hits, dict) and "results" in hits:
                hits = hits["results"]
            return hits[:k] if isinstance(hits, list) else []
        except Exception as e:
            print(f"WARN: Memory search failed: {e}")
            return []

memory = Mem0ManagedAdapter(MEM0_API)

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

app = FastAPI(
    title="Tableau Chatbot Agent API",
    description="An API that uses a LangChain agent to interact with Tableau for visualization + Q&A",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LLM selection ----------------
def make_llm():
    """Pick an LLM based on available keys. Low temp for tool-use reliability."""
    if ChatGoogleGenerativeAI is not None:
        try:
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
        except Exception:
            pass
    if ChatOpenAI is not None:
        try:
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception:
            pass
    raise RuntimeError(
        "No LLM configured. Set GOOGLE_API_KEY for Gemini or OPENAI_API_KEY for OpenAI."
    )

llm = make_llm()

# -------------- Prompt (hub w/ local fallback) --------------
try:
    prompt = hub.pull("hwchase17/react")
except Exception:
    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate.from_template(
        """You are a helpful Tableau assistant.

You can use these tools:
{tools}

Allowed tool names: {tool_names}

Guidelines:
- If the user wants to SEE a chart, call `tableau_get_view_image`.
- If the user wants to ANALYZE a chart/table, call `tableau_get_view_data`, then answer from those rows.
- If asked to refresh a datasource, call `refresh_datasource_now`.
- Do NOT print raw data URLs or base64 in your final answer.
- When you fetch an image, just acknowledge it briefly; the UI will display it.
- ALWAYS prefer using a tool over guessing.
- If filters are mentioned (e.g., Region=APAC, Year=2024), pass them as JSON to filters_json.

Use a ReAct loop:
1) Think about what to do.
2) Action: <tool name> with an Action Input
3) Observation: tool output
Repeat 1–3 as needed, then finish with a concise Final Answer.

Question: {input}

{agent_scratchpad}
"""
    )

# -------------- Tools wiring --------------
tools = [
    list_tableau_projects,
    list_tableau_workbooks,
    list_tableau_views,
    tableau_get_view_image,
    tableau_get_view_data,
    publish_mock_datasource,
    refresh_datasource_now,
    create_hourly_refresh_schedule,
    tableau_job_status,
]

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

# -------------- Schemas --------------
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = USER_ID  # Use environment USER_ID as default

class ChatResponse(BaseModel):
    response: str
    attachments: Optional[List[Dict[str, Any]]] = None

DATA_URL_RE = re.compile(
    r'(data:image/(?:png|jpeg|jpg|gif|webp);base64,[A-Za-z0-9+/=]+)'
)

# -------------- Helpers --------------
def _extract_attachments(intermediate_steps):
    attachments = []
    for _action, observation in intermediate_steps:
        if not observation:
            continue
        s = observation if isinstance(observation, str) else str(observation)
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
        m = DATA_URL_RE.search(s)
        if m:
            attachments.append({"type": "image", "dataUrl": m.group(1), "caption": ""})
    return attachments

# -------------- Routes --------------
@app.get("/")
def read_root():
    return {"status": "Tableau Chatbot Agent is running!"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Free-form natural language. The agent decides which Tableau tool(s) to call.
    Integrates Mem0 for memory storage and retrieval with personal slot resolution.
    """
    # Check if the query is a question
    def is_question(raw: str) -> bool:
        tq = raw.strip().lower()
        return (
            "?" in raw
            or tq.startswith(("what ", "whats", "what's", "who ", "where ", "when ", "why ", "how ",
                              "do i ", "am i ", "is my ", "are my "))
            or tq.startswith(("list ", "find ", "show ", "recommend ", "suggest ", "give me ", "tell me ", "name"))
        )

    if is_question(request.message):
        # Try memory first for questions
        mems = memory.search(request.user_id, request.message, k=5)
        chosen = pick_relevant_memory(request.message, mems or [], min_score=0.55)
        if chosen:
            val = extract_value_from_memory(request.message, chosen)
            if val:
                # Save the interaction
                memory.save(request.user_id, request.message, {"kind": "question"})
                memory.save(request.user_id, val, {"kind": "answer"})
                return {"response": val, "attachments": []}
        
        # Resolve personal slots for context
        augmented, resolved = resolve_personal_slots(request.message, lambda q, k=5: memory.search(request.user_id, q, k))
        input_query = augmented if resolved else request.message
    else:
        # Non-question: save to Mem0 and acknowledge
        memory.save(request.user_id, request.message, {"kind": "utterance"})
        memory.save(request.user_id, "Noted. I've saved that.", {"kind": "answer"})
        return {"response": "Noted. I've saved that.", "attachments": []}

    # Use LangChain agent for Tableau-related or unresolved queries
    result = agent_executor.invoke({"input": input_query})
    attachments = _extract_attachments(result.get("intermediate_steps", []))
    raw_text = result.get("output", "Sorry, I couldn't produce a response.")
    output_text = _clean_output_text(raw_text, attachments)

    # Save the interaction
    memory.save(request.user_id, request.message, {"kind": "question"})
    memory.save(request.user_id, output_text, {"kind": "answer"})

    return {"response": output_text, "attachments": attachments}

# Optional: direct REST endpoints
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

@app.post("/datasources/schedule")
def api_schedule(ds: str = "AI_Sample_Sales", schedule_name: str = "AI-Hourly-Demo"):
    return {"message": create_hourly_refresh_schedule.invoke({"datasource_name": ds, "schedule_name": schedule_name})}

@app.post("/datasources/refresh-now")
def api_refresh_now(ds: str = "AI_Sample_Sales"):
    return {"message": refresh_datasource_now.invoke({"datasource_name": ds})}