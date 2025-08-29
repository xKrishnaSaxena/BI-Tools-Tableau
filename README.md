# Tableau Chatbot Agent (FastAPI + React)

A minimal full-stack app that lets you chat with your Tableau Server. The backend uses a LangChain ReAct agent wired to Tableau Server Client (TSC) tools to fetch view **images** or **data**, publish a mock datasource, and manage extract refreshes. The frontend is a lightweight chat UI that renders returned images/tables and lets you download tables as CSV.

---

## 📽️ Walkthrough

[walkthrough](https://github.com/user-attachments/assets/b68ea83c-083b-40e0-b6fc-673055beedfd)

---

## Features

- 💬 Natural-language chat to **list projects/workbooks/views**
- 🖼️ “Show me the chart” → fetches **high-res PNG** of a Tableau view
- 📊 “Analyze the table” → fetches **CSV as JSON rows** for in-chat analysis
- ⚙️ **Refresh** a datasource now; **create hourly schedules**; **check job status**
- 🔧 Works with **Gemini** (preferred) or **OpenAI** as the LLM (auto-fallback)
- 🔐 Uses Tableau **Personal Access Token** (PAT)

---

## Architecture

```
Frontend (React)
   |
   |  POST /chat   (free-form message)
   v
Backend (FastAPI + LangChain ReAct Agent)
   - Tool: list_tableau_projects
   - Tool: list_tableau_workbooks
   - Tool: list_tableau_views
   - Tool: tableau_get_view_image
   - Tool: tableau_get_view_data
   - Tool: publish_mock_datasource
   - Tool: refresh_datasource_now
   - Tool: create_hourly_refresh_schedule
   - Tool: tableau_job_status
   |
   v
Tableau Server (TSC SDK)
```

The backend extracts **attachments** (images/tables) from tool outputs and returns them separately so the UI can render cleanly.

---

## Prerequisites

- **Python** 3.10+
- **Node.js** 18+
- Access to a **Tableau Server/Cloud** site with a PAT
- At least one LLM key:

  - Google: `GOOGLE_API_KEY`
  - or OpenAI: `OPENAI_API_KEY`

---

## Environment Variables

Create a `.env` in the backend root:

```ini
# Tableau
TABLEAU_SERVER_URL=https://your-tableau-server
TABLEAU_SITE_CONTENT_URL=your_site_content_url   # blank for Default
TABLEAU_PAT_NAME=your_pat_name
TABLEAU_PAT_SECRET=your_pat_secret

# LLMs (prefer Google; fallback to OpenAI)
GOOGLE_API_KEY=your_google_key
OPENAI_API_KEY=your_openai_key
```

> The agent prefers **Gemini** (`gemini-2.0-flash-lite`) via `langchain_google_genai`; if not available it falls back to **OpenAI** (`gpt-4o-mini`) via `langchain_openai`.

---

## Quick Start

### 1) Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# If you don't have one, install minimal deps:
# pip install fastapi uvicorn "langchain>=0.2" python-dotenv \
#   langchain-google-genai langchain-openai tableauserverclient pandas pantab
uvicorn main:app --reload --port 8000
```

You should see: `{"status": "Tableau Chatbot Agent is running!"}` at `http://127.0.0.1:8000/`.

### 2) Frontend

```bash
cd frontend
npm install
# If this is a Vite app:
npm run dev
# If this is Create React App:
# npm start
```

Open the dev server (usually printed by your tooling, e.g. `http://localhost:5173` or `http://localhost:3000`).

> The UI calls the backend at `http://127.0.0.1:8000/chat`. CORS is enabled for all origins in dev.

---

## Using the App

Type messages like:

- “List Tableau projects”
- “What workbooks do I have?”
- “Show view **Sales Overview** from workbook **Superstore** with **Region=APAC, Year=2024**”
- “Get the data for **Profit by Category** (limit 200 rows)”
- “Refresh datasource **AI_Sample_Sales** now”
- “Create an hourly schedule for **AI_Sample_Sales**”
- “Check job status **\<job_id>**”

The UI will:

- Render **images** the agent fetches (PNG data URLs)
- Render **tables** and offer **Download CSV**

---

## API (Backend)

### `GET /`

Health check.

### `POST /chat`

Free-form chat. The agent decides which Tableau tool(s) to call.

**Request**

```json
{ "message": "Show view Sales Overview with Region=APAC" }
```

**Response**

```json
{
  "response": "Rendered view 'Sales Overview'.",
  "attachments": [
    {
      "type": "image",
      "dataUrl": "data:image/png;base64,...",
      "caption": "Rendered view 'Sales Overview'."
    }
  ]
}
```

### `GET /views/image`

Return PNG (as data URL) for a view.

**Query params**

- `view_name` (required)
- `workbook_name` (optional)
- `filters_json` (optional JSON string, e.g. `{"Region":"APAC","Year":"2024"}`)

**Example**

```
/views/image?view_name=Sales%20Overview&workbook_name=Superstore&filters_json=%7B%22Region%22%3A%22APAC%22%7D
```

### `GET /views/data`

Return view data as JSON rows (from CSV).

**Query params**

- `view_name` (required)
- `workbook_name` (optional)
- `filters_json` (optional)
- `max_rows` (default 200)

### `POST /datasources/publish-mock`

Publishes a small **.hyper** mock datasource (`AI_Sample_Sales`) to project `AI Demos` (creates if missing).

### `POST /datasources/schedule`

Create (or reuse) an **hourly extract schedule** and attach a datasource.

**Query params**

- `ds` (default `AI_Sample_Sales`)
- `schedule_name` (default `AI-Hourly-Demo`)

### `POST /datasources/refresh-now`

Kick off an extract refresh immediately.

**Query params**

- `ds` (default `AI_Sample_Sales`)

---

## How It Works

- **LLM selection:** `make_llm()` prefers Gemini (`langchain_google_genai`), falls back to OpenAI (`langchain_openai`), both with `temperature=0` for reliable tool usage.
- **Agent prompt:** ReAct-style. If user wants to **see** a chart → use `tableau_get_view_image`. If they want to **analyze** → `tableau_get_view_data`.
- **Attachments:** The backend scrapes tool outputs for `{ "image": "data:image/png;base64,..." }` or `{ "table": [...] }`. The frontend renders them. Base64 strings are **not** shown in text (only displayed as images).
- **Filters:** Pass filters as JSON (e.g., `{"Region":"APAC","Year":"2024"}`) to `filters_json`. These become `vf` parameters for TSC requests.
- **CSV handling:** `tableau_get_view_data` robustly handles bytes/stream payloads, decodes with `utf-8-sig`, and returns `columns` + `table` rows.

---

## Frontend Notes

- Simple chat UI with bubbles; shows bot/user avatars.
- **Data tables** render with a compact, scrollable grid.
- **Download CSV** button exports the table the agent returned.
- Update `fetch("http://127.0.0.1:8000/chat")` in `App.jsx` if your backend runs elsewhere.

---

## Troubleshooting

- **“No LLM configured”**
  Ensure `GOOGLE_API_KEY` or `OPENAI_API_KEY` is in `.env` and the corresponding package is installed.

- **Tableau auth errors**
  Check `TABLEAU_SERVER_URL`, `TABLEAU_SITE_CONTENT_URL`, `TABLEAU_PAT_NAME`, `TABLEAU_PAT_SECRET`. PAT must have access to the site/project.

- **CORS / network**
  Frontend must reach `http://127.0.0.1:8000`. Change the URL in `App.jsx` if needed. CORS is wide-open in dev; tighten for prod.

- **View or workbook not found**
  Verify names and (optionally) specify `workbook_name`. Use “List views/workbooks” to discover exact names.

---
