# backend/main.py
import json
from typing import List, Dict, Any, Optional

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
import re
JSON_CODEBLOCK_RE = re.compile(r"```(?:json|csv|table)?[\s\S]*?```", re.IGNORECASE)
def _clean_output_text(text: str, attachments) -> str:
    if not isinstance(text, str):
        return ""

    # Remove "Final Answer:" noise
    text = text.replace("Final Answer:", "").strip()

    # Strip any data URLs that might slip in
    text = re.sub(r'data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=\s]+', '', text).strip()

    # Drop fenced code blocks like ```json ... ```
    text = JSON_CODEBLOCK_RE.sub("", text).strip()

    # If we have a table attachment, hide raw JSON/CSV dumps in the text
    has_table = any(a.get("type") == "table" for a in (attachments or []))
    if has_table:
        t = text.lstrip()
        if (t.startswith("[") or t.startswith("{")) or len(t) > 300:
            text = ""

    # If nothing left, fall back to the attachment caption (or a generic line)
    if attachments and not text:
        for a in attachments:
            if a.get("caption"):
                return a["caption"]
        return "Here’s your data."

    return text

load_dotenv()

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

# -------------- Prompt (hub w/ local fallback) --------------
try:
    prompt = hub.pull("hwchase17/react")
except Exception:
    # Minimal local ReAct-style prompt
    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate.from_template(
        """You are a helpful Tableau assistant. Use the tools to get images or data from Tableau.
If the user wants to SEE a chart, call `tableau_get_view_image`.
If the user wants to ANALYZE a chart/table, call `tableau_get_view_data`, then answer from those rows.
If asked to refresh a datasource, call `refresh_datasource_now`.
Do NOT print raw data URLs or base64 in your final answer. 
When you fetch an image, just acknowledge it briefly; the UI will display it.
ALWAYS prefer using a tool over guessing. If filters are mentioned (e.g., Region=APAC, Year=2024), pass them as JSON to filters_json.

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


class ChatResponse(BaseModel):
    response: str
    attachments: Optional[List[Dict[str, Any]]] = None  # [{"type":"image"/"table","..."}]

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

        # 1) Prefer strict JSON from tools like {"text": "...", "image": "data:image/..."}
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

        # 2) Fallback: extract ONLY the data URL substring from any noisy text
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
    The response includes lifted attachments (image/table) when available.
    """
    result = agent_executor.invoke({"input": request.message})
    attachments = _extract_attachments(result.get("intermediate_steps", []))
    raw_text = result.get("output", "Sorry, I couldn't produce a response.")
    output_text = _clean_output_text(raw_text, attachments)
    return {"response": output_text, "attachments": attachments}


# Optional: direct REST endpoints if you want to bypass the agent from the UI.
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

