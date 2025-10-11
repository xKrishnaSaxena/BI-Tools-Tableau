from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
from fastapi.responses import HTMLResponse
from graph_runner import build_graph
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

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"  # scope memories/conversation

class ChatResponse(BaseModel):
    response: str
    attachments: Optional[List[Dict[str, Any]]] = None
GRAPH = build_graph()

def _dedupe_attachments(attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for a in attachments:
        key = json.dumps({k: a.get(k) for k in ("type","name","id","url","meta")}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id or "default"
    init_state = {
    "session_id": session_id,
    "user_text": request.message.strip(),
    "messages": [],  # Already resets, but confirm
    "tool_steps": [],  # Explicitly reset to avoid accumulation
    "attachments": [],  # Reset
    "recalled": [],  # Optional: if you want fresh recall each time
}

    out = await GRAPH.ainvoke(
    init_state,
    config={"configurable": {"thread_id": session_id, "langgraph_user_id": session_id}},
)
    final_text = out.get("final_text", "")
    attachments = _dedupe_attachments(out.get("attachments", []))

    tool_steps = out.get("tool_steps", [])
    if tool_steps:
        latest_turn = max(ts.get("turn_idx", 0) for ts in tool_steps if isinstance(ts, dict))
        attachments = [a for a in attachments if a.get("turn_idx", latest_turn) == latest_turn]

    return ChatResponse(response=final_text, attachments=attachments)

@app.get("/graph3", response_class=HTMLResponse)
def get_graph_diagram():
    mermaid_source = GRAPH.get_graph().draw_mermaid()

    html_content = f"""
    <html>
    <head>
      <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    </head>
    <body>
      <div class="mermaid">
      {mermaid_source}
      </div>
      <script>
        mermaid.initialize({{ startOnLoad: true }});
      </script>
    </body>
    </html>
    """

    return html_content