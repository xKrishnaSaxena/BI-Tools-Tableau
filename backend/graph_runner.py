from __future__ import annotations
from typing import Dict, Any, List, Optional, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver  
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
import os
from collections import defaultdict
from pydantic import BaseModel, Field, field_validator
import logging
from dotenv import load_dotenv
from enum import Enum
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import json
from langmem import create_memory_store_manager  
from langgraph.store.memory import InMemoryStore
from langgraph.graph.message import add_messages
import uuid
import csv
from tab_tools import (
    list_tableau_projects,
    list_tableau_workbooks,
    list_tableau_views,
    tableau_get_view_image,
    tableau_get_view_data,
    publish_mock_datasource,
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted,DeadlineExceeded  # Import for retry_if
import pandas as pd  
from io import StringIO
# Add this decorator to LLM functions
@retry(
    retry=retry_if_exception_type((ResourceExhausted, DeadlineExceeded)),
    stop=stop_after_attempt(2),  # Reduce from 3 to 2
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Shorter waits
    reraise=True
)
def safe_llm_invoke(llm, messages):
    return llm.invoke(messages)

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

def make_llm(api_key=None, model_name="gemini-2.5-flash-lite"):
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("langchain_google_genai not available.")
    try:
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("No API key provided.")
        log.info("Using Gemini chat model: %s with API key", model_name)
        return ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.2, 
            google_api_key=api_key,
            timeout=30.0,  # Add timeout
            max_retries=2   # Limit retries
        )  # Slight temp for naturalness
    except Exception as e:
        log.error("Failed to init Gemini chat model: %s", e, exc_info=True)
        raise

def make_llm_parsing():
    api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        temperature=0, 
        google_api_key=api_key,
        timeout=15.0,  # Shorter timeout for parsing
        max_retries=1
    )  # Temp 0 for deterministic parsing

def make_llm_2_0_publishing():
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("langchain_google_genai not available.")
    try:
        api_key = os.getenv("GOOGLE_API_KEY_2_0")
        if not api_key:
            raise ValueError("No GOOGLE_API_KEY_2_0 provided for publishing operations.")
        log.info("Using Gemini 2.0 Flash Lite for publishing operations")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite", 
            temperature=0, 
            google_api_key=api_key
        )
    except Exception as e:
        log.error("Failed to init Gemini 2.0 chat model for publishing: %s", e, exc_info=True)
        raise

# Global LLMs
llm_2_5 = make_llm()  # Default to 2.5 with primary API key
llm_parsing = make_llm_parsing()  # Dedicated for parsing with temp=0
llm_publishing = make_llm_2_0_publishing()  # Dedicated publishing LLM
llm = llm_2_5

TOOLS_PUBLISH_ONLY = [publish_mock_datasource]
TOOLS = [
    list_tableau_projects,
    list_tableau_workbooks,
    list_tableau_views,
    tableau_get_view_image,
    tableau_get_view_data,
    publish_mock_datasource,
]
TOOL_REGISTRY = {t.name: t for t in TOOLS}
DEFAULTS_BY_SESSION: Dict[str, Dict[str, Any]] = defaultdict(dict)
class DefaultPrefs(BaseModel):
    project_name: Optional[str] = None
    datasource_name: Optional[str] = None
    workbook_name: Optional[str] = None
    view_name: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
class DefaultsActionType(str, Enum):
    NONE = "none"
    SET = "set_defaults"
    APPLY = "apply_defaults"
    CLEAR = "clear_defaults"

class DefaultsAction(BaseModel):
    action: DefaultsActionType = DefaultsActionType.NONE
    defaults: Optional[DefaultPrefs] = None
    reason: Optional[str] = None
    missing_keys: Optional[List[str]] = None

class Intent(str, Enum):
    VIEW_IMAGE = "view_image"               
    VIEW_DATA = "view_data"                 
    PUBLISH_MOCK = "publish_mock_datasource"
    MEMORY_QA = "memory_qa"           
    MEMORY_SET = "memory_set"     
    HELP = "help"                          
    SMALLTALK = "smalltalk"                 
    LIST_PROJECTS = "list_projects"          
    LIST_WORKBOOKS = "list_workbooks"       
    LIST_VIEWS = "list_views"                
    GENERAL_QA = "general_qa"  # New intent for general questions
    UNKNOWN = "unknown"

class PublishArgs(BaseModel):
    project_name: Optional[str] = None
    datasource_name: Optional[str] = None

class PostProcessAnswer(BaseModel):
    answer: str
    suggest_next_tool: bool = False
    next_tool_name: Optional[str] = None
    next_tool_args: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None

class ParsedCommand(BaseModel):
    intent: Intent = Field(..., description="Best intent for this user message.")
    memory_fact: Optional[Dict[str, str]] = None
    view_name: Optional[str] = None
    workbook_name: Optional[str] = None
    filters_json: Optional[Dict[str, Any]] = None   
    
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
    @field_validator("memory_fact", mode="before")
    @classmethod
    def _memory_fact_coerce(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
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
    is_specific_memory_question: Optional[bool] = False
    reason: str = Field(..., description="Short rationale of why this intent/slots were chosen.")
def _merge_defaults(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (new or {}).items():
        if k == "filters" and isinstance(v, dict):
            out.setdefault("filters", {})
            out["filters"].update(v)
        elif v is not None:
            out[k] = v
    return out

def _llm_infer_defaults_from_snippets(recalled: List[str]) -> Dict[str, Any]:
    if not recalled:
        return {}
    # Prioritize filter-related snippets (e.g., "south region only as thats our market scope")
    filter_snippets = [s for s in recalled if any(kw in s.lower() for kw in ['filter', 'region', 'south', 'market scope', 'only'])]
    all_snippets = filter_snippets + [s for s in recalled if s not in filter_snippets]  # Filters first
    
    sys = SystemMessage(content=(
        "From the following conversation snippets, infer the user's CURRENT defaults.\n"
        "Only include fields you are confident about: project_name, datasource_name, workbook_name, view_name, filters (object).\n"
        "Prioritize **filters** if mentioned (e.g., {'Region': 'South'} from 'south region only').\n"
        "Prefer the most recent information if there are conflicts. If nothing concrete, return an empty object."
    ))
    prompt = "Snippets (most recent last, filters prioritized):\n- " + "\n- ".join(all_snippets or [])
    try:
        structured = llm.with_structured_output(DefaultPrefs)
        prefs: DefaultPrefs = structured.invoke([sys, HumanMessage(content=prompt)])
        inferred = prefs.model_dump(exclude_none=True)
        if inferred.get('filters'):
            log.info("Inferred filters from snippets: %s", inferred['filters'])
        return inferred
    except Exception:
        return {}

def _llm_understand_defaults_command(user_text: str, recalled: List[str]) -> DefaultsAction:
    sys = SystemMessage(content=(
        "You are a precise interpreter for defaults management commands.\n"
        "Classify the user's message into one of: set_defaults, apply_defaults, clear_defaults, none.\n"
        "When setting, extract ONLY fields explicitly implied: project_name, datasource_name, workbook_name, view_name, filters (object).\n"
        "For filters, prefer a JSON object with concrete key-value pairs. If ambiguous (e.g., just 'Furniture'), only set it if the likely key is obvious from context (e.g., Category). Otherwise, omit.\n"
        "Never invent values; be conservative.\n"
        "Use conversation snippets to understand references like 'use defaults', 'same as before', or previously stated default names.\n"
        "If the user says 'always filter/view/show only X' for a view, set defaults.view_name, defaults.workbook_name, and defaults.filters with inferred keys (e.g., {'Region': 'South'} if obvious like 'south region').\n"
        "Context keys like 'region', 'category' are common; use them if implied."
    ))
    memories = "\n".join(f"- {m}" for m in (recalled or []))
    prompt = (
        f"User message:\n{user_text}\n\n"
        f"Relevant prior snippets (most recent last):\n{memories or '- (none)'}\n\n"
        "Return structured output."
    )
    try:
        structured = llm.with_structured_output(DefaultsAction)
        return structured.invoke([sys, HumanMessage(content=prompt)])
    except Exception:
        return DefaultsAction(action=DefaultsActionType.NONE)

def _llm_parse_user_utterance(user_text: str, recalled_snippets: List[str], messages: List[AnyMessage] = []) -> ParsedCommand:
    # --- System prompt (includes Fix 4 content update) ---
    sys = SystemMessage(content=(
        "You are a friendly data analyst assistant specializing in Tableau. Classify the user's intent and extract arguments precisely.\n"
        "- Be natural and infer like a colleague would.\n"
        "- CRITICAL FOR PUBLISH_MOCK: If the recent conversation shows the assistant asked for project/datasource names, "
        "treat ANY user response with names (even just 'X and Y' or 'project is X, datasource is Y') as intent=PUBLISH_MOCK.\n"
        "- For PUBLISH_MOCK, ALWAYS populate publish.project_name and publish.datasource_name from the user's text.\n"
        "- Look for patterns like: 'project name is X', 'use Y for datasource', 'X and Y', 'project: A, datasource: B'\n"
        "- Map verbs:\n"
        "  * SEE/visualize → VIEW_IMAGE (e.g., 'show me', 'display', 'render image', 'preview').\n"
        "  * DATA/analysis → VIEW_DATA (e.g., 'export csv', 'get data', 'analyze table', 'download rows').\n"
        "  * If both, prefer VIEW_DATA if data export is mentioned.\n"
        "- For filters (e.g., 'only South Region in 2024'), normalize to filters_json like {'Region': 'South', 'Year': '2024'}.\n"
        "- If asking to recall facts ('what is my name?', 'where do I work?'), set intent=MEMORY_QA and is_specific_memory_question=True.\n"
        "- If asking about the conversation ('what are we talking about?', 'what happened so far?'), set intent=MEMORY_QA and is_specific_memory_question=True.\n"
        "- If stating facts ('my name is X', 'I work in Y'), even without 'remember', set intent=MEMORY_SET and memory_fact={'name': 'X'} or similar.\n"
        "- Do NOT confuse project/datasource names with view/workbook names—PUBLISH_MOCK is about creating datasources, not views.\n"
        "- If requesting to publish mock data, set intent=PUBLISH_MOCK and fill publish fields.\n"
        "- For help/capabilities, set HELP.\n"
        "- For greetings/small talk, set SMALLTALK.\n"
        "- For listing ('list projects', 'what workbooks are there?'), set LIST_* accordingly.\n"
        "- For general questions about Tableau or data analysis (e.g., 'What is Tableau?'), set GENERAL_QA.\n"
        "- If unclear, set UNKNOWN.\n"
        "- Use snippets for context; if snippets show the assistant asked for project/datasource names, treat the user's reply as filling those for PUBLISH_MOCK.\n"
        "Examples:\n"
        "User: project name is sam and datasource name is kamal\n"
        "Parsed: intent=PUBLISH_MOCK, publish={\"project_name\": \"sam\", \"datasource_name\": \"kamal\"}\n"
        "User: sam and kamal (when context shows we asked for publish names)\n"
        "Parsed: intent=PUBLISH_MOCK, publish={\"project_name\": \"sam\", \"datasource_name\": \"kamal\"}\n"
    ))

    recent_history = "\n".join([f"{msg.type}: {str(msg.content)[:200]}" for msg in messages[-6:]])
    snippet_block = ""
    if recalled_snippets:
        top = recalled_snippets[:6]
        snippet_block += "Relevant snippets (most recent last):\n- " + "\n- ".join(top) + "\n"
    if recent_history:
        snippet_block += f"\nRecent conversation history:\n{recent_history}\n"

    structured_llm = llm_parsing.with_structured_output(ParsedCommand)
    hm = HumanMessage(content=(
        f"{snippet_block}"
        "Return ONLY fields defined by the schema; avoid made-up keys.\n\n"
        f"User message:\n{user_text}"
    ))

    try:
        parsed: ParsedCommand = safe_llm_invoke(structured_llm, [sys, hm])
        log.info(f"Parsed intent for '{user_text}': {parsed.intent}, reason: {parsed.reason}")

        if isinstance(parsed.filters_json, str):
            try:
                fj = json.loads(parsed.filters_json)
                parsed.filters_json = fj if isinstance(fj, dict) else None
            except Exception:
                parsed.filters_json = None

        # ENHANCED: Post-parsing fix for PUBLISH_MOCK
        if parsed.intent == Intent.PUBLISH_MOCK:
            # If publish is None or has default/empty values
            if parsed.publish is None or not parsed.publish.project_name or not parsed.publish.datasource_name:
                # Check if this is a follow-up response to a request for names
                if _is_publish_followup(messages, recalled_snippets, user_text):
                    # Extract names from the user's message
                    extracted = _extract_project_datasource_names(user_text)
                    if extracted['project_name'] or extracted['datasource_name']:
                        parsed.publish = PublishArgs(
                            project_name=extracted['project_name'] or "AI Demos",
                            datasource_name=extracted['datasource_name'] or "AI_Sample_Sales"
                        )
                        log.info(f"Post-parse fix applied: publish={parsed.publish}")

                # Fallback: Check if misparsed to workbook/view fields
                elif parsed.workbook_name or parsed.view_name:
                    parsed.publish = PublishArgs(
                        project_name=parsed.workbook_name or "AI Demos",
                        datasource_name=parsed.view_name or "AI_Sample_Sales"
                    )
                    parsed.workbook_name = None
                    parsed.view_name = None
                    log.info(f"Post-parse reassignment: publish={parsed.publish}")

        # If still not PUBLISH_MOCK but context suggests it should be
        elif parsed.intent in (Intent.UNKNOWN, Intent.SMALLTALK) and _is_publish_followup(messages, recalled_snippets, user_text):
            extracted = _extract_project_datasource_names(user_text)
            if extracted['project_name'] or extracted['datasource_name']:
                parsed.intent = Intent.PUBLISH_MOCK
                parsed.publish = PublishArgs(
                    project_name=extracted['project_name'] or "AI Demos",
                    datasource_name=extracted['datasource_name'] or "AI_Sample_Sales"
                )
                parsed.reason = "Corrected to PUBLISH_MOCK based on context"
                log.info(f"Intent corrected to PUBLISH_MOCK: publish={parsed.publish}")

        return parsed
    except Exception as e:
        log.warning("LLM parse failed, falling back to UNKNOWN: %s", e, exc_info=True)
        return ParsedCommand(intent=Intent.UNKNOWN, reason="LLM parse error")

def get_session_defaults(session_id: str, recalled: List[str]) -> Dict[str, Any]:
    stored = dict(DEFAULTS_BY_SESSION.get(session_id, {}))
    if stored:  # Skip infer if we have stored defaults
        return stored
    inferred = _llm_infer_defaults_from_snippets(recalled)
    return _merge_defaults(inferred, stored)

def apply_defaults_to_parsed(
    parsed: ParsedCommand,
    defaults: Dict[str, Any],
    apply_now: bool,
) -> ParsedCommand:
    if parsed.intent == Intent.PUBLISH_MOCK:
        if parsed.publish:
            if (apply_now or not parsed.publish.project_name) and defaults.get("project_name"):
                parsed.publish.project_name = parsed.publish.project_name or defaults["project_name"]
            if (apply_now or not parsed.publish.datasource_name) and defaults.get("datasource_name"):
                parsed.publish.datasource_name = parsed.publish.datasource_name or defaults["datasource_name"]

    if parsed.intent in (Intent.VIEW_IMAGE, Intent.VIEW_DATA):
        if (apply_now or not parsed.workbook_name) and defaults.get("workbook_name"):
            parsed.workbook_name = parsed.workbook_name or defaults["workbook_name"]
        if (apply_now or not parsed.view_name) and defaults.get("view_name"):
            parsed.view_name = parsed.view_name or defaults["view_name"]

        f_defaults = defaults.get("filters") if isinstance(defaults.get("filters"), dict) else {}
        f_explicit = parsed.filters_json or {}
        merged = dict(f_defaults or {})
        merged.update(f_explicit or {})
        if merged:  # Log if applying inferred filters
            log.info("Applied merged filters: %s (explicit: %s, default: %s)", merged, f_explicit, f_defaults)
        parsed.filters_json = merged or None

    return parsed

def _is_publish_followup(messages: List[AnyMessage], recalled_snippets: List[str], user_text: str) -> bool:
    """
    Detect if this message is a follow-up response to a request for project/datasource names.
    """
    # Check last 4 assistant messages for asking about publish info
    try:
        recent_assistant_msgs = [
            (msg.content or "").lower() for msg in messages[-4:]
            if isinstance(msg, AIMessage)
        ]
    except Exception:
        recent_assistant_msgs = []

    # Keywords that indicate we asked for publish info
    publish_ask_patterns = [
        'what project', 'which project', 'what datasource', 'which datasource',
        'what names', 'project and datasource', 'project name', 'datasource name',
        'should i use', 'mock data', 'publish'
    ]

    # Check if assistant recently asked about publishing
    for msg in recent_assistant_msgs:
        if any(pattern in msg for pattern in publish_ask_patterns):
            log.info(f"Detected publish follow-up context from assistant msg: {msg[:100]}")
            return True

    # Check recalled snippets for publish context (recent conversation)
    snippet_text = ' '.join((recalled_snippets or [])[-10:]).lower()
    if 'publish' in snippet_text or 'mock' in snippet_text:
        user_lower = (user_text or '').lower()
        # User is likely providing name-like info (not just casual chat)
        name_indicators = ['project', 'datasource', 'name', 'is', 'use', 'for', 'and', 'call']
        if any(indicator in user_lower for indicator in name_indicators):
            log.info("Detected publish follow-up from snippets + user text patterns")
            return True

    return False


def _extract_project_datasource_names(text: str) -> Dict[str, Optional[str]]:
    """
    Extract project and datasource names from user text using LLM.
    This handles ANY naming pattern flexibly.
    """
    result = {'project_name': None, 'datasource_name': None}

    try:
        sys = SystemMessage(content=(
            "Extract EXACTLY the project name and datasource name from the user's message.\n"
            "The user may phrase it in many ways:\n"
            "- 'project name is X and datasource name is Y'\n"
            "- 'use X for project, Y for datasource'\n"
            "- 'X and Y' (when asking about publishing)\n"
            "- 'project: X, datasource: Y'\n"
            "- 'call it X and Y'\n"
            "- any other natural phrasing\n\n"
            "Be flexible but precise. Extract the actual names, preserving exact spelling, spaces, and capitalization.\n"
            "If only one name is clear, populate that field. If neither is clear, return nulls."
        ))
        prompt = HumanMessage(content=f"User message: {text}\n\nExtract names:")

        class NameExtraction(BaseModel):
            project_name: Optional[str] = None
            datasource_name: Optional[str] = None

        structured = llm_parsing.with_structured_output(NameExtraction)
        extraction: NameExtraction = safe_llm_invoke(structured, [sys, prompt])

        if extraction.project_name:
            result['project_name'] = extraction.project_name.strip()
        if extraction.datasource_name:
            result['datasource_name'] = extraction.datasource_name.strip()

        log.info(f"LLM extracted from '{(text or '')[:50]}...': {result}")
    except Exception as e:
        log.warning(f"Name extraction failed: {e}")

    return result

class Filters(BaseModel):
    filters: Optional[Dict[str, Any]] = None

def _extract_filters_from_text(text: str, recalled: List[str]) -> Optional[Dict[str, Any]]:
    sys = SystemMessage(content=(
        "Extract filters from the user's response as a dict like {'Region': 'South', 'Year': 2024}. "
        "Use common Tableau field names like Region, Category, Sales, Order Date, etc. "
        "If unclear or no filters mentioned, return empty dict. Be conservative and only include explicit key-value pairs."
    ))
    context = "\n".join(recalled[-3:]) if recalled else ""
    prompt = HumanMessage(content=f"Context: {context}\nUser filters request: {text}")
    try:
        structured = llm_parsing.with_structured_output(Filters)
        res: Filters = safe_llm_invoke(structured, [sys, prompt])
        return res.filters or {}
    except Exception as e:
        log.warning(f"Filter extraction failed: {e}")
        return {}

# ---------------- Memory layer ----------------
def _truncate(s: str, n: int = 200) -> str:
    return (s[: n - 3] + "...") if s and len(s) > n else (s or "")

USER_TAIL_MAX = int(os.getenv("USER_TAIL_MAX", "600"))
RECALL_SEM_K = int(os.getenv("RECALL_SEM_K", "48"))
RECALL_TAIL_K = int(os.getenv("RECALL_TAIL_K", "48"))
RECALL_TOTAL_K = int(os.getenv("RECALL_TOTAL_K", "48"))
MEMORY_Q_TAIL_K = int(os.getenv("MEMORY_Q_TAIL_K", "64")) 

class UserTailMemory:
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
            gem_model = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004").strip()
            if not gem_model.startswith("models/"):
                gem_model = f"models/{gem_model}"

            embedder = GoogleGenerativeAIEmbeddings(
                model=gem_model,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            dims = int(os.getenv("EMBED_DIMS", "768"))
            
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
            self.manager = create_memory_store_manager(model, store=self.store)
            self.available = True
            log.info("LangMem initialized and available.")
        except Exception as e:
            log.error("Failed to initialize LangMem: %s", e, exc_info=True)
            self.available = False

    def add(self, session_id: str, text: str) -> bool:
        if not self.available or not self.manager:
            log.debug("LangMem.add skipped (unavailable)."); return False
        try:
            cfg = {"configurable": {"langgraph_user_id": session_id}}
            self.manager.invoke({"messages": [{"role": "user", "content": text}]}, config=cfg)
            log.info("LangMem.add stored OK: session=%s", session_id)
            return True
        except Exception as e:
            log.error("LangMem.add failed: session=%s err=%s", session_id, e, exc_info=True)
            return False

    def search(self, session_id: str, query: str, k: int = RECALL_SEM_K) -> List[str]:
        if not self.available or not self.manager:
            log.debug("LangMem.search skipped (unavailable)."); return []
        try:
            cfg = {"configurable": {"langgraph_user_id": session_id}}
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
            return out[:k]
        except Exception as e:
            log.error("LangMem.search failed: session=%s err=%s", session_id, e, exc_info=True)
            return []
USER_TAIL = UserTailMemory()
LANGMEM = LangMemWrapper(llm)
def pin_fact(session_id: str, text: str):
    if not text:
        return
    try:
        stored = LANGMEM.add(session_id, f"[FACT] {text}")
        if stored:
            log.debug("Pinned fact to LangMem: %s", _truncate(text))
        else:
            log.debug("Skipped pin (LangMem unavailable).")
    except Exception as e:
        log.warning("Failed to pin fact: %s", e)

def _summarize_tool_context(tool_steps: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    blobs = []
    for step in tool_steps or []:
        name = step.get("name", "")
        args = step.get("args", {})
        raw = step.get("result", "")
        s = raw if isinstance(raw, str) else str(raw)
        # Strip large data URLs
        if 'data:image/' in s:
            s = s.replace(s[s.find('data:image/'):s.find('==')+2], '[image omitted]')
        s = s.strip()
        if len(s) > 2000:
            s = s[:2000] + " …(truncated)"
        blobs.append(f"- TOOL {name}\n  ARGS: {json.dumps(args, ensure_ascii=False)}\n  RESULT: {s}")
    text = "\n".join(blobs)
    return text[:max_chars]

def _llm_post_process_answer(
    user_text: str,
    recalled: List[str],
    tool_steps: List[Dict[str, Any]],
    llm_obj=None,
) -> Optional[str]:
    if not tool_steps:
        return None

    sys = SystemMessage(content=(
        "You are a friendly data analyst employee. Respond in a natural, professional tone, like chatting with a colleague.\n"
        "Use ONLY the provided tool outputs and recalled memories to craft your answer.\n"
        "If the user mentioned 'my/our <something>', resolve from memories.\n"
        "Filter or reshape results if needed (e.g., 'projects with my name'), but NEVER invent data.\n"
        "If insufficient, say so politely and suggest one next step if relevant."
    ))

    memories = "\n".join(f"- {m}" for m in (recalled or []))
    tools_ctx = _summarize_tool_context(tool_steps)

    prompt = (
        f"User question:\n{user_text}\n\n"
        f"Recalled memory snippets:\n{memories or '- (none)'}\n\n"
        f"Tool results:\n{tools_ctx or '- (none)'}\n\n"
        "Craft a helpful, concise response. Use bullets for lists. Be engaging and human-like."
    )

    structured = (llm_obj or llm).with_structured_output(PostProcessAnswer)
    try:
        resp: PostProcessAnswer = safe_llm_invoke(structured, [sys, HumanMessage(content=prompt)])
    except Exception:
        raw = safe_llm_invoke(llm, [sys, HumanMessage(content=prompt)])
        return (raw.content or "").strip()

    text = (resp.answer or "").strip()
    if not text:
        return None

    if resp.suggest_next_tool and resp.next_tool_name:
        tip = f"\n\nIf you'd like, I can try {resp.next_tool_name} next with these details: {json.dumps(resp.next_tool_args or {}, ensure_ascii=False)}."
        text += tip

    return text

def memory_add(session_id: str, role: str, text: str):
    cleaned_text = (text or "").strip()
    if not cleaned_text: return
    if role == "user":
        LANGMEM.add(session_id, cleaned_text)
        USER_TAIL.add_user(session_id, cleaned_text)
    else:
        LANGMEM.add(session_id, f"[ASSISTANT] {cleaned_text}")

def memory_recall(session_id: str, query: str) -> List[str]:
    augmented_query = f"{query} recent conversation assistant response project datasource names"
    sem = LANGMEM.search(session_id, augmented_query, k=RECALL_SEM_K) or []
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
def gather_memory_snippets(session_id: str, query: str, extra_k: int = 10) -> List[str]:
    qs = [
        query,
        f"User self info: {query}",
        f"My details: {query}",
        f"Earlier you said: {query}",
    ]
    out: List[str] = []
    seen = set()
    for q in qs:
        hits = LANGMEM.search(session_id, q, k=extra_k) or []
        for h in hits:
            if h and h not in seen:
                out.append(h); seen.add(h)
    for t in USER_TAIL.last_n(session_id, n=MEMORY_Q_TAIL_K):
        if t and t not in seen:
            out.append(t); seen.add(t)
    return out[:24]

def _supported_by_snippets(ans: str, snippets: List[str]) -> bool:
    if not ans: return False
    al = ans.lower()
    parts = [p.strip() for p in al.split() if len(p) >= 3]  # Word-level check instead of regex
    haystack = " ".join(snippets).lower()
    return any(p in haystack for p in parts)

def _summarize_history(messages: List[AnyMessage], max_len: int = 2000) -> str:
    """Summarize conversation history for context."""
    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[-10:]])  # Last 10 messages
    if len(history_text) <= max_len:
        return history_text
    # Use LLM to summarize if too long
    sys = SystemMessage(content="Summarize the following conversation history concisely.")
    prompt = HumanMessage(content=history_text)
    resp = safe_llm_invoke(llm, [sys, prompt])
    return resp.content.strip()

def memory_qa(state: AgentState) -> AgentState:
    session_id = state["session_id"]
    user_text = state["user_text"]
    snippets = gather_memory_snippets(session_id, user_text)
    history_summary = _summarize_history(state.get("messages", []))  # Add history summary
    if not snippets and not history_summary:
        return {"final_text": "Sorry, I don't recall that right now. Could you remind me?"}

    sys = (
        "You are a friendly data analyst. Answer using ONLY the snippets and history below. "
        "Be natural and helpful, like 'From what I remember, your name is...'. "
        "If the answer isn't clear, politely say you don't recall."
    )
    prompt = (
        f"Snippets:\n- " + "\n- ".join(snippets) + "\n\n"
        f"Conversation history summary:\n{history_summary or '- (none)'}\n\n"
        f"Question: {user_text}\n\n"
        "Respond conversationally:"
    )
    try:
        resp: AIMessage = safe_llm_invoke(llm, [SystemMessage(content=sys), HumanMessage(content=prompt)])
        ans = (resp.content or "").strip()
        if not ans or not _supported_by_snippets(ans, snippets + [history_summary]):
            ans = "Hmm, I couldn't find that in my notes. Want to tell me again so I can jot it down?"
        if 'my ' in ans.lower():
            ans = ans.replace('my ', 'your ').replace('My ', 'Your ')
        return {"final_text": ans}
    except Exception:
        return {"final_text": "Sorry, something went wrong while checking my notes. Let's try that again."}

# New node for general QA
def general_qa(state: AgentState) -> AgentState:
    user_text = state["user_text"]
    history_summary = _summarize_history(state.get("messages", []))  # Include history for context

    sys = SystemMessage(content=(
        "You are an expert Tableau data analyst. Answer general questions about Tableau, data visualization, "
        "analytics, dashboards, etc., in a helpful, professional tone. Use your knowledge base. "
        "Keep responses concise and accurate. If the question relates to the conversation, reference the history."
    ))
    prompt = (
        f"Conversation history summary (for context):\n{history_summary or '- (none)'}\n\n"
        f"User question: {user_text}\n\n"
        "Provide a clear, informative answer:"
    )
    try:
        resp: AIMessage = safe_llm_invoke(llm, [sys, HumanMessage(content=prompt)])
        ans = (resp.content or "").strip()
        return {"final_text": ans}
    except Exception:
        return {"final_text": "Sorry, I encountered an issue answering that. Let's try rephrasing."}

# ---------------- Cleaning / attachments ----------------
def _clean_output_text(text: str, attachments) -> str:
    if not isinstance(text, str): return ""
    text = text.replace("Final Answer:", "").strip()
    if 'data:image/' in text:
        text = text[:text.find('data:image/')] + '[image omitted]' + text[text.rfind('==')+2:]
    if '```' in text:
        start = text.find('```')
        end = text.rfind('```') + 3
        text = text[:start] + text[end:]
    text = text.strip()
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
            return "Here's the CSV file with your data export."
        return "Here's the data you requested."
    return text

def _write_csv(rows: list, columns: list, prefix: str = "tableau_export") -> str:
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
    for step in tool_steps[-1:]:  # Only latest step to avoid concat
        s = step.get("result")
        if not s: continue
        s = s if isinstance(s, str) else str(s)
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                img = obj.get("image")
                if isinstance(img, str) and 'data:image/' in img:
                    attachments.append({"type": "image", "dataUrl": img, "caption": obj.get("text", "")})
                tbl = obj.get("table")
                if isinstance(tbl, list) and tbl:
                    cols = obj.get("columns", [])
                    caption = obj.get("text", "")
                    # NEW: If table is full, but we have filters in step args, filter here too (redundancy)
                    filters_json = json.loads(step.get("args", {}).get("filters_json", "{}"))
                    if filters_json:
                        try:
                            df = pd.DataFrame(tbl)
                            for col, val in filters_json.items():
                                if col in df.columns:
                                    df = df[df[col].astype(str) == str(val)]
                            tbl = df.to_dict('records')
                            cols = list(df.columns) if df.columns.tolist() != cols else cols
                            log.info("Attachment table filtered to %d rows", len(tbl))
                        except Exception:
                            pass
                    attachments.append({
                        "type": "table",
                        "rows": tbl,
                        "columns": cols,
                        "caption": caption,
                    })
                    csv_path = _write_csv(tbl, cols)
                    attachments.append({
                        "type": "file",
                        "mime": "text/csv",
                        "path": csv_path,
                        "filename": os.path.basename(csv_path),
                        "caption": caption or "Exported data (filtered)",
                    })
                continue
        except Exception:
            pass
        if 'data:image/' in s:
            start = s.find('data:image/')
            end = s.find('==', start) + 2
            attachments.append({"type": "image", "dataUrl": s[start:end], "caption": ""})
    return attachments  # Already deduped via latest step

class AgentState(TypedDict, total=False):
    session_id: str
    messages: Annotated[List[AnyMessage], add_messages]   
    user_text: str

    recalled: Annotated[List[str], operator.add]          
    defaults_cmd: DefaultsAction
    session_defaults: Dict[str, Any]
    apply_now: bool

    parsed: ParsedCommand
    route: str

    tool_steps: List[Dict[str, Any]] 
    attachments: List[Dict[str, Any]]

    final_text: str
    error: str

    awaiting_filters: bool
    pending_view: Dict[str, Any]

# 1) Ingest user message
def ingest(state: AgentState) -> AgentState:
    text = state["user_text"].strip()
    return {
        "messages": [HumanMessage(content=text)],
        # reset transient/turn-scoped fields
        "route": None,
        "final_text": "",
        "tool_steps": [],  # Already reset, but ensure
        "attachments": [],  # Explicitly reset to prevent concat
        "defaults_cmd": DefaultsAction(),  # optional
        "parsed": None,                    # optional
    }

# 2) Recall snippets
def recall(state: AgentState) -> AgentState:
    session_id = state["session_id"]
    text = state["user_text"]
    primary = memory_recall(session_id, text)
    extra = gather_memory_snippets(session_id, text, extra_k=6)
    seen = set()
    merged = []
    for s in primary + extra:
        if s not in seen:
            merged.append(s); seen.add(s)
    return {"recalled": merged}

# 3) Parse user intent + defaults command
def parse_and_defaults_cmd(state: AgentState) -> AgentState:
    text = state["user_text"]
    recalled = state.get("recalled", [])
    messages = state.get("messages", [])
    parsed = _llm_parse_user_utterance(text, recalled,messages)
    defaults_cmd = _llm_understand_defaults_command(text, recalled)

    session_id = state["session_id"]
    if defaults_cmd.action == DefaultsActionType.SET and defaults_cmd.defaults:
        vals = defaults_cmd.defaults.model_dump(exclude_none=True)
        DEFAULTS_BY_SESSION[session_id] = _merge_defaults(DEFAULTS_BY_SESSION.get(session_id, {}), vals)
        try:
            LANGMEM.add(session_id, f"[DEFAULTS] {json.dumps(vals)}")
        except Exception:
            pass

    if defaults_cmd.action == DefaultsActionType.CLEAR:
        DEFAULTS_BY_SESSION.pop(session_id, None)
        try:
            LANGMEM.add(session_id, "[DEFAULTS] cleared")
        except Exception:
            pass
        return {
            "final_text": "Got it, I've cleared your saved defaults for this session.",
            "route": "finalize"
        }

    return {
        "parsed": parsed,
        "defaults_cmd": defaults_cmd,
    }

# 4) Apply defaults
def apply_defaults_node(state: AgentState) -> AgentState:
    session_id = state["session_id"]
    parsed: ParsedCommand = state["parsed"]
    defaults_cmd: DefaultsAction = state.get("defaults_cmd") or DefaultsAction()
    apply_now = (defaults_cmd.action == DefaultsActionType.APPLY)
    session_defaults = get_session_defaults(session_id, state.get("recalled", []))
    parsed = apply_defaults_to_parsed(parsed, session_defaults, apply_now)

    return {
        "parsed": parsed,
        "session_defaults": session_defaults,
        "apply_now": apply_now,
    }

# 5) Router
def router(state: AgentState) -> AgentState:
    parsed: ParsedCommand = state["parsed"]
    session_id = state["session_id"]
    recalled = state.get("recalled", [])
    print(parsed)
    print(session_id)
    print(recalled)

    # Handle awaiting filters case
    if state.get("awaiting_filters", False):
        user_lower = state["user_text"].lower().strip()
        decline_phrases = ["no", "none", "no thanks", "without filters", "all", "unfiltered", "default", "no filter"]
        if any(phrase in user_lower for phrase in decline_phrases):
            pending = state.get("pending_view", {})
            parsed = ParsedCommand(
                intent=pending.get("intent", Intent.VIEW_DATA),
                view_name=pending.get("view_name"),
                workbook_name=pending.get("workbook_name"),
                filters_json=None,
                reason="User chose no filters"
            )
            return {
                "parsed": parsed,
                "awaiting_filters": False,
                "pending_view": {},
                "route": "call_tool_view"
            }
        else:
            # Extract filters from user text
            filters = _extract_filters_from_text(state["user_text"], recalled)
            pending = state.get("pending_view", {})
            parsed = ParsedCommand(
                intent=pending.get("intent", Intent.VIEW_DATA),
                view_name=pending.get("view_name"),
                workbook_name=pending.get("workbook_name"),
                filters_json=filters,
                reason="Extracted filters from user response"
            )
            return {
                "parsed": parsed,
                "awaiting_filters": False,
                "pending_view": {},
                "route": "call_tool_view"
            }

    if parsed.intent == Intent.MEMORY_SET:
        fact = parsed.memory_fact or {}
        if fact:
            key, val = next(iter(fact.items()), ("fact", "unknown"))
            pin_fact(session_id, f"my {key} is {val}")
            return {
                "final_text": f"Sure thing, I've noted that your {key} is {val}.",
                "route": "finalize"
            }
        else:
            return {"route": "finalize"}
    if parsed.intent == Intent.MEMORY_QA:
        return {"route": "memory_qa"}
    if parsed.intent == Intent.SMALLTALK:
        previous_assistant = LANGMEM.search(session_id, "assistant greeting response hello hi welcome", k=5)  # Query words likely to match stored [ASSISTANT] content
        previous_str = ', '.join([
            m.split(']', 1)[1].strip() if ']' in m else m 
            for m in previous_assistant if '[ASSISTANT]' in m
        ]) or 'none'

        sys = SystemMessage(content=(
            "Generate a friendly greeting for a data analyst chatbot, varying it slightly each time. "
            "Make sure to vary the greeting to avoid repetition if previous ones exist. Keep it short and inviting."
        ))  

        prompt = HumanMessage(content=f"Previous greetings if any: {previous_str} \nUser: {state['user_text']}")

        resp = safe_llm_invoke(llm, [sys, prompt])
        greeting = resp.content.strip() or "Hey! Good to hear from you. What data insights can I help with today?"
        return {
            "final_text": greeting,
            "route": "finalize"
        }
    if parsed.intent == Intent.HELP:
        return {
            "final_text": "I'm your go-to for Tableau tasks—like listing projects, workbooks, or views; pulling images or data from views (with filters if needed); publishing mock datasources; or even remembering details about you. Just let me know what you need!",
            "route": "finalize"
        }
    if parsed.intent == Intent.GENERAL_QA:  # New route
        return {"route": "general_qa"}
    if parsed.intent == Intent.PUBLISH_MOCK:
        pub = parsed.publish or PublishArgs()
        pn = (pub.project_name or "").strip()
        dn = (pub.datasource_name or "").strip()
    if parsed.intent == Intent.PUBLISH_MOCK:
        pub = parsed.publish or PublishArgs()
        pn = (pub.project_name or "").strip()
        dn = (pub.datasource_name or "").strip()

        # Improved validation: Also check for default placeholder values
        missing_or_default = (
            not pn or not dn or 
            pn.lower() in ("ai demos", "aidemos", "default") or 
            dn.lower() in ("ai_sample_sales", "aisamplesales", "default", "sample")
        )

        if missing_or_default:
            log.warning(f"PUBLISH_MOCK needs names: pn='{pn}', dn='{dn}'")
            # Store a hint in session defaults that we're waiting for publish args
            DEFAULTS_BY_SESSION[session_id]['_awaiting_publish'] = True
            return {
                "final_text": "No problem—what project and datasource names should I use for that?",
                "route": "finalize"
            }

        # Clear the awaiting flag if it was set
        DEFAULTS_BY_SESSION[session_id].pop('_awaiting_publish', None)
        return {"route": "call_tool_publish"}

    if parsed.intent in (Intent.LIST_PROJECTS, Intent.LIST_WORKBOOKS, Intent.LIST_VIEWS):
        return {"route": "call_tool_list"}

    if parsed.intent in (Intent.VIEW_IMAGE, Intent.VIEW_DATA):
        # NEW: If inferred/merged filters exist, skip prompt and apply directly
        if parsed.filters_json:
            log.info("Skipping filter prompt; applying inferred filters: %s", parsed.filters_json)
            return {"route": "call_tool_view"}
        else:
            return {
                "awaiting_filters": True,
                "pending_view": {
                    "intent": parsed.intent,
                    "view_name": parsed.view_name,
                    "workbook_name": parsed.workbook_name
                },
                "final_text": f"Got it—you want to {'see an image of' if parsed.intent == Intent.VIEW_IMAGE else 'get data from'} the '{parsed.view_name}' view{ ' in workbook ' + parsed.workbook_name if parsed.workbook_name else '' }. Do you want to apply any filters (e.g., 'Region: South' or 'Sales > 10000')? Reply 'no' if not.",
                "route": "finalize"
            }

    # Unknown: polite fallback
    return {
        "final_text": "I'm not quite sure what you meant there. Could you clarify? For example, try 'show me the Sales Overview' or 'export data from Profit by Category'.",
        "route": "finalize"
    }

# 6a) Call listing tools
def call_tool_list(state: AgentState) -> AgentState:
    parsed: ParsedCommand = state["parsed"]
    tool_steps = []

    if parsed.intent == Intent.LIST_PROJECTS:
        name = "list_tableau_projects"; args = {}
    elif parsed.intent == Intent.LIST_WORKBOOKS:
        name = "list_tableau_workbooks"; args = {}
    else:
        name = "list_tableau_views"
        args = {}
        if parsed.workbook_name:
            args["workbook_name"] = parsed.workbook_name

    try:
        result = TOOL_REGISTRY[name].invoke(args)
    except Exception as e:
        result = f"Tool '{name}' failed: {e}"
        log.error("Tool call failed: %s", e)

    step = {"name": name, "args": args, "result": result}
    return {"tool_steps": [step]}

# 6b) Call view tools
def call_tool_view(state: AgentState) -> AgentState:
    parsed: ParsedCommand = state["parsed"]
    if parsed.intent == Intent.VIEW_IMAGE:
        name = "tableau_get_view_image"
    else:
        name = "tableau_get_view_data"

    args = {
        "view_name": parsed.view_name or "",
        "workbook_name": parsed.workbook_name or "",
        "filters_json": json.dumps(parsed.filters_json or {}),
    }
    if not args["view_name"]:
        return {
            "final_text": "Sure, which view are you thinking of? For example, 'Sales Overview'.",
            "route": "finalize"
        }

    try:
        result = TOOL_REGISTRY[name].invoke(args)
    except Exception as e:
        result = f"Tool '{name}' failed: {e}"
        log.error("Tool call failed: %s", e)

    # NEW: For VIEW_DATA, apply client-side filter if filters provided (fallback if tool ignores)
    if parsed.intent == Intent.VIEW_DATA and parsed.filters_json:
        try:
            # Assume result is JSON with 'table' (list of dicts) or CSV string
            if isinstance(result, str) and result.startswith('{'):
                obj = json.loads(result)
                table = obj.get('table', [])
                if table and isinstance(table[0], dict):  # List of dicts
                    df = pd.DataFrame(table)
                    for col, val in parsed.filters_json.items():
                        if col in df.columns:
                            df = df[df[col] == val]  # Exact match; extend for >/< if needed
                            log.info("Client-side filtered %s rows on %s=%s", len(df), col, val)
                    obj['table'] = df.to_dict('records')
                    result = json.dumps(obj)
                elif 'Returned' in result and '\t' in result:  # TSV/CSV string
                    # Parse as TSV (from logs: tab-separated)
                    lines = result.split('\n')
                    if len(lines) > 1:
                        df = pd.read_csv(StringIO('\n'.join(lines)), sep='\t')
                        for col, val in parsed.filters_json.items():
                            if col in df.columns:
                                df = df[df[col].astype(str) == str(val)]
                                log.info("Client-side filtered %s rows on %s=%s", len(df), col, val)
                        # Reconstruct TSV
                        tsv_str = df.to_csv(sep='\t', index=False)
                        result = result.split('\n')[0] + '\n' + tsv_str  # Preserve header like "Returned X rows"
            log.info("Post-tool filter applied successfully")
        except Exception as fe:
            log.warning("Client-side filter failed: %s", fe)

    step = {"name": name, "args": args, "result": result}
    state["tool_steps"] = [step]  # Ensure single step

    # NEW: If VIEW_DATA succeeded with filters, persist as default
    if parsed.intent == Intent.VIEW_DATA and parsed.filters_json:
        session_id = state["session_id"]
        DEFAULTS_BY_SESSION[session_id] = _merge_defaults(DEFAULTS_BY_SESSION.get(session_id, {}), {'filters': parsed.filters_json})
        LANGMEM.add(session_id, f"[DEFAULTS] Persistent filter for {parsed.view_name}: {json.dumps(parsed.filters_json)} (market scope)")
        log.info("Persisted filters as default: %s", parsed.filters_json)

    return {"tool_steps": [step]}

# 6c) Call publish tool
def call_tool_publish(state: AgentState) -> AgentState:
    pub = (state["parsed"].publish) or PublishArgs()
    args = {"project_name": pub.project_name, "datasource_name": pub.datasource_name}
    try:
        result = TOOL_REGISTRY["publish_mock_datasource"].invoke(args)
    except Exception as e:
        result = f"Tool 'publish_mock_datasource' failed: {e}"
        log.error("Tool call failed: %s", e)

    step = {"name": "publish_mock_datasource", "args": args, "result": result}
    return {"tool_steps": [step]}

# 7) Post-process
def postprocess(state: AgentState) -> AgentState:
    tool_steps = state.get("tool_steps", [])
    recalled = state.get("recalled", [])
    text = state["user_text"]

    attachments = _extract_attachments(tool_steps)
    pp = _llm_post_process_answer(text, recalled, tool_steps, llm_obj=llm)

    final_text = pp or ""
    if not final_text and tool_steps:
        for step in tool_steps:
            s = str(step.get("result", "")).strip()
            if not s:
                continue
            if s.startswith("{") and s.endswith("}"):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                        final_text = obj["text"]; break
                except Exception:
                    pass
            final_text = s; break

    final_text = _clean_output_text(final_text, attachments)
    if final_text and attachments:
        final_text = f"{final_text}\n\nI've attached the visuals/data below for you."
    return {"attachments": attachments, "final_text": final_text}

# 8) Finalize + store memories
def finalize_and_store(state: AgentState) -> AgentState:
    session_id = state["session_id"]
    user_text = state["user_text"]
    final_text = state.get("final_text", "") or "All set!"
    # Append assistant response to messages
    state["messages"].append(AIMessage(content=final_text))

    memory_add(session_id, "user", user_text)
    memory_add(session_id, "assistant", final_text)
    return {"final_text": final_text}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("ingest", ingest)
    graph.add_node("recall", recall)
    graph.add_node("parse_and_defaults_cmd", parse_and_defaults_cmd)
    graph.add_node("apply_defaults", apply_defaults_node)
    graph.add_node("router", router)
    graph.add_node("memory_qa", memory_qa)
    graph.add_node("general_qa", general_qa)  # New node
    graph.add_node("call_tool_list", call_tool_list)
    graph.add_node("call_tool_view", call_tool_view)
    graph.add_node("call_tool_publish", call_tool_publish)
    graph.add_node("postprocess", postprocess)
    graph.add_node("finalize", finalize_and_store)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "recall")
    graph.add_edge("recall", "parse_and_defaults_cmd")

    def _after_parse(state: AgentState):
        if state.get("route") == "finalize" and state.get("final_text"): 
            return "finalize"
        return "apply_defaults"
    graph.add_conditional_edges("parse_and_defaults_cmd", _after_parse, {
        "finalize": "finalize",
        "apply_defaults": "apply_defaults",
    })

    graph.add_edge("apply_defaults", "router")

    def _route_decider(state: AgentState):
        return state.get("route", "finalize")
    graph.add_conditional_edges("router", _route_decider, {
        "memory_qa": "memory_qa",
        "general_qa": "general_qa",  # New
        "call_tool_list": "call_tool_list",
        "call_tool_view": "call_tool_view",
        "call_tool_publish": "call_tool_publish",
        "finalize": "finalize",
    })

    graph.add_edge("memory_qa", "finalize")
    graph.add_edge("general_qa", "finalize")  # New
    graph.add_edge("call_tool_list", "postprocess")
    graph.add_edge("call_tool_view", "postprocess")
    graph.add_edge("call_tool_publish", "postprocess")
    graph.add_edge("postprocess", "finalize")

    checkpointer=MemorySaver()
    app_graph=graph.compile(checkpointer=checkpointer)
    return app_graph