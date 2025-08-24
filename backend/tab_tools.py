
import os, io, csv, json, base64, tempfile
from typing import Optional,Any
from dotenv import load_dotenv
import tableauserverclient as TSC
import pandas as pd
import pantab as pt 
from langchain.tools import tool 
load_dotenv()

TABLEAU_SERVER_URL = (os.getenv("TABLEAU_SERVER_URL", "") or "").rstrip("/")
SITE = os.getenv("TABLEAU_SITE_CONTENT_URL", "") 
PAT_NAME = os.getenv("TABLEAU_PAT_NAME", "")
PAT_SECRET = os.getenv("TABLEAU_PAT_SECRET", "")

def _server() -> TSC.Server:
    if not TABLEAU_SERVER_URL or not PAT_NAME or not PAT_SECRET:
        raise RuntimeError("Missing Tableau env vars (SERVER_URL, PAT_NAME, PAT_SECRET).")
    auth = TSC.PersonalAccessTokenAuth(PAT_NAME, PAT_SECRET, site_id=SITE)
    server = TSC.Server(TABLEAU_SERVER_URL, use_server_version=True)
    server.auth.sign_in(auth)
    return server

def _coerce_one(value: Any, *keys: str) -> Any:
    """Accept raw strings like 'key=\"val\"' or '{\"key\":\"val\"}' and return 'val'."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    # JSON object?
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            for k in keys:
                if k in obj:
                    v = obj[k]
                    return v.strip("'\"") if isinstance(v, str) else v
        except Exception:
            pass
    # key=value form
    for k in keys:
        prefix = f"{k}="
        if s.lower().startswith(prefix.lower()):
            return s[len(prefix):].strip().strip("'\"")
    # strip surrounding quotes
    return s.strip("'\"")


# put near your other helpers
def _bytes_from_csv_payload(payload) -> bytes:
    """Return raw bytes from TSC CSV payload (bytes, file-like, or generator)."""
    if payload is None:
        return b""
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if hasattr(payload, "read"):                # e.g., BytesIO / file-like
        return payload.read()
    try:                                        # generator / iterable of bytes
        return b"".join(chunk for chunk in payload)
    except TypeError:
        return str(payload).encode("utf-8", "ignore")



# ---------- Helpers ----------
def _find_project_id(server: TSC.Server, name: str) -> Optional[str]:
    projects, _ = server.projects.get()
    for p in projects:
        if p.name.lower() == name.lower():
            return p.id
    return None

def _find_or_create_project(server: TSC.Server, name: str) -> str:
    pid = _find_project_id(server, name)
    if pid:
        return pid
    proj = TSC.ProjectItem(name=name)
    proj = server.projects.create(proj)
    return proj.id

def _find_workbook(server: TSC.Server, workbook_name: str) -> Optional[TSC.WorkbookItem]:
    wbs, _ = server.workbooks.get()
    for wb in wbs:
        if wb.name.lower() == workbook_name.lower():
            return wb
    return None

def _find_view(server: TSC.Server, view_name: str, workbook_name: Optional[str]) -> Optional[TSC.ViewItem]:
    if workbook_name:
        wb = _find_workbook(server, workbook_name)
        if not wb:
            return None
        server.workbooks.populate_views(wb)
        for v in wb.views:
            if v.name.lower() == view_name.lower():
                return v
        return None
    # search site-wide
    views, _ = server.views.get()
    for v in views:
        if v.name.lower() == view_name.lower():
            return v
    return None

def _find_datasource(server: TSC.Server, datasource_name: str) -> Optional[TSC.DatasourceItem]:
    dss, _ = server.datasources.get()
    for ds in dss:
        if ds.name.lower() == datasource_name.lower():
            return ds
    return None

# ---------- READS ----------
@tool
def list_tableau_projects() -> str:
    """List all Tableau projects you have access to."""
    try:
        server = _server()
        projs, _ = server.projects.get()
        server.auth.sign_out()
        if not projs:
            return "No projects found."
        return "Projects: " + ", ".join(p.name for p in projs)
    except Exception as e:
        return f"Error listing projects: {e}"

@tool
def list_tableau_workbooks() -> str:
    """List workbooks visible to you."""
    try:
        server = _server()
        wbs, _ = server.workbooks.get()
        server.auth.sign_out()
        if not wbs:
            return "No workbooks found."
        return "Workbooks: " + ", ".join(w.name for w in wbs)
    except Exception as e:
        return f"Error listing workbooks: {e}"

@tool
def list_tableau_views(workbook_name: str = "") -> str:
    """List views; optionally pass a workbook_name to narrow the list."""
    try:
        workbook_name = _coerce_one(workbook_name, "workbook_name")
        server = _server()
        if workbook_name:
            wb = _find_workbook(server, workbook_name)
            if not wb:
                server.auth.sign_out()
                return f"Workbook '{workbook_name}' not found."
            server.workbooks.populate_views(wb)
            names = ", ".join(v.name for v in (wb.views or []))
            server.auth.sign_out()
            return f"Views in '{workbook_name}': {names or 'None'}"
        else:
            views, _ = server.views.get()
            names = ", ".join(v.name for v in views)
            server.auth.sign_out()
            return f"Views: {names or 'None'}"
    except Exception as e:
        return f"Error listing views: {e}"
    
import re, json

def _parse_react_kv(s: str) -> dict:
    """
    Parse strings like 'view_name=Overview, workbook_name=Superstore, filters_json={"Region":"APAC"}'
    into a dict. Leaves non 'k=v' tokens alone.
    """
    out = {}
    if not isinstance(s, str) or "=" not in s:
        return out
    # split on commas that are not inside braces
    parts = re.split(r',\s*(?![^{}]*\})', s)
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out

@tool
def tableau_get_view_image(view_name: str, workbook_name: str = "", filters_json: str = "") -> str:
    """Return a PNG (data URL) for a view. Optional filters_json like '{"Region":"APAC"}'."""
    try:
        if isinstance(view_name, str) and ("=" in view_name or view_name.strip().startswith("{")):
            kv = _parse_react_kv(view_name)
            view_name = kv.get("view_name", view_name)
            # allow workbook_name/filters_json to come from the combined string if not provided
            workbook_name = kv.get("workbook_name", workbook_name)
            filters_json = kv.get("filters_json", filters_json)

        # Final cleanup/coercion
        view_name = _coerce_one(view_name, "view_name")
        workbook_name = _coerce_one(workbook_name, "workbook_name")
        filters_json = _coerce_one(filters_json, "filters_json")
        filt = json.loads(filters_json) if filters_json else {}
        server = _server()
        v = _find_view(server, view_name, workbook_name or None)
        if not v:
            server.auth.sign_out()
            return json.dumps({"text": f"View '{view_name}' not found.", "image": None})
        opts = TSC.ImageRequestOptions(
            imageresolution=TSC.ImageRequestOptions.Resolution.High, maxage=1
        )
        for k, val in filt.items():
            opts.vf(k, str(val))
        server.views.populate_image(v, opts)
        img_b64 = base64.b64encode(v.image).decode("utf-8")
        server.auth.sign_out()
        return json.dumps({"text": f"Rendered view '{v.name}'.", "image": f"data:image/png;base64,{img_b64}"})
    except Exception as e:
        return json.dumps({"text": f"Image fetch failed: {e}", "image": None})
    

@tool
def tableau_get_view_data(view_name: str, workbook_name: str = "", filters_json: str = "", max_rows: int = 200) -> str:
    """Download a view’s data as CSV and return JSON rows (up to max_rows)."""
    try:
        if isinstance(view_name, str) and ("=" in view_name or view_name.strip().startswith("{")):
            kv = _parse_react_kv(view_name)
            view_name = kv.get("view_name", view_name)
            # allow workbook_name/filters_json to come from the combined string if not provided
            workbook_name = kv.get("workbook_name", workbook_name)
            filters_json = kv.get("filters_json", filters_json)
        view_name = _coerce_one(view_name, "view_name")
        workbook_name = _coerce_one(workbook_name, "workbook_name")
        filters_json = _coerce_one(filters_json, "filters_json")
        filt = json.loads(filters_json) if filters_json else {}

        server = _server()
        v = _find_view(server, view_name, workbook_name or None)
        if not v:
            server.auth.sign_out()
            return json.dumps({"text": f"View '{view_name}' not found.", "table": []})

        opts = TSC.CSVRequestOptions(maxage=1)
        for k, val in filt.items():
            opts.vf(k, str(val))

        server.views.populate_csv(v, opts)

        # -------- FIX: handle generator/stream/bytes --------
        raw = _bytes_from_csv_payload(v.csv)
        # use utf-8-sig to gracefully drop a BOM if present
        text = raw.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        headers = reader.fieldnames or []
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)

        server.auth.sign_out()
        return json.dumps({
            "text": f"Returned {len(rows)} rows from '{v.name}'.",
            "columns": headers,
            "table": rows
        })
    except Exception as e:
        try:
            server.auth.sign_out()
        except Exception:
            pass
        return json.dumps({"text": f"Data fetch failed: {e}", "table": []})

# ---------- WRITES ----------
@tool
def publish_mock_datasource(project_name: str = "AI Demos", datasource_name: str = "AI_Sample_Sales", overwrite: bool = False) -> str:
    """Create a small mock dataset, write to .hyper, and publish a datasource."""
    try:
        if isinstance(project_name, str) and ("=" in project_name or project_name.strip().startswith("{")):
            kv = _parse_react_kv(project_name)
            project_name = kv.get("project_name", project_name)
            datasource_name = kv.get("datasource_name", datasource_name)
            ow = kv.get("overwrite", overwrite)
            if isinstance(ow, str):
                overwrite = ow.strip().lower() in ("1", "true", "yes", "y")
            elif isinstance(ow, bool):
                overwrite = ow

        if isinstance(datasource_name, str) and ("=" in datasource_name or datasource_name.strip().startswith("{")):
            kv = _parse_react_kv(datasource_name)
            datasource_name = kv.get("datasource_name", datasource_name)
            project_name = kv.get("project_name", project_name)
            ow = kv.get("overwrite", overwrite)
            if isinstance(ow, str):
                overwrite = ow.strip().lower() in ("1", "true", "yes", "y")
            elif isinstance(ow, bool):
                overwrite = ow

        project_name = _coerce_one(project_name, "project_name")
        datasource_name = _coerce_one(datasource_name, "datasource_name")

        import tempfile, os
        hyper_path = os.path.join(tempfile.gettempdir(), f"{datasource_name}.hyper")
        df = pd.DataFrame(
            [
                ["2024-01-05", "APAC", "Technology", "Headphones", 2, 200.0],
                ["2024-02-11", "EMEA", "Office Supplies", "Paper", 50, 75.0],
                ["2024-03-22", "Americas", "Furniture", "Chair", 1, 300.0],
                ["2024-04-03", "APAC", "Furniture", "Desk", 1, 450.0],
                ["2024-04-17", "EMEA", "Technology", "Keyboard", 3, 120.0],
            ],
            columns=["OrderDate", "Region", "Category", "Product", "Quantity", "Sales"],
        )

        # Windows-safe temp path
        hyper_path = os.path.join(tempfile.gettempdir(), f"{datasource_name}.hyper")
        pt.frame_to_hyper(df, hyper_path, table="Extract.Extract")

        server = _server()
        proj_id = _find_or_create_project(server, project_name or "AI Demos")
        ds_item = TSC.DatasourceItem(project_id=proj_id, name=datasource_name)
        mode = TSC.Server.PublishMode.Overwrite if overwrite else TSC.Server.PublishMode.CreateNew
        published = server.datasources.publish(ds_item, hyper_path, mode)
        server.auth.sign_out()
        return f"Published datasource '{published.name}' (id={published.id}) in project '{project_name or 'AI Demos'}'."
    except Exception as e:
        return f"Publish failed: {e}"

@tool
def refresh_datasource_now(datasource_name: str) -> str:
    """Kick off an extract refresh immediately and return the Job ID."""
    try:
        if isinstance(datasource_name, str) and ("=" in datasource_name or datasource_name.strip().startswith("{")):
            kv = _parse_react_kv(datasource_name)
            datasource_name = kv.get("datasource_name", datasource_name)
        datasource_name = _coerce_one(datasource_name, "datasource_name")
        server = _server()
        ds = _find_datasource(server, datasource_name)
        if not ds:
            server.auth.sign_out()
            return f"Datasource '{datasource_name}' not found."
        job = server.datasources.refresh(ds)
        job_id = getattr(job, "id", None)
        server.auth.sign_out()
        return f"Started refresh for '{datasource_name}'. Job ID: {job_id}"
    except Exception as e:
        return f"Refresh failed: {e}"

@tool
def create_hourly_refresh_schedule(datasource_name: str, schedule_name: str = "AI-Hourly-Demo") -> str:
    """Create (or reuse) an hourly extract schedule and attach the datasource."""
    from datetime import time as dtime
    try:
        if isinstance(datasource_name, str) and ("=" in datasource_name or datasource_name.strip().startswith("{")):
            kv = _parse_react_kv(datasource_name)
            datasource_name = kv.get("datasource_name", datasource_name)
            schedule_name = kv.get("schedule_name", schedule_name)

        if isinstance(schedule_name, str) and ("=" in schedule_name or schedule_name.strip().startswith("{")):
            kv = _parse_react_kv(schedule_name)
            schedule_name = kv.get("schedule_name", schedule_name)
            datasource_name = kv.get("datasource_name", datasource_name)
        datasource_name = _coerce_one(datasource_name, "datasource_name")
        schedule_name = _coerce_one(schedule_name, "schedule_name")
        server = _server()
        ds = _find_datasource(server, datasource_name)
        if not ds:
            server.auth.sign_out()
            return f"Datasource '{datasource_name}' not found."

        schedules = {s.name: s for s in server.schedules.get()[0]}
        if schedule_name in schedules:
            sched = schedules[schedule_name]
        else:
            hourly = TSC.HourlyInterval(start_time=dtime(0, 0), end_time=dtime(23, 59), interval_value=1)
            sched = TSC.ScheduleItem(
                name=schedule_name,
                priority=60,
                schedule_type=TSC.ScheduleItem.Type.Extract,
                execution_order=TSC.ScheduleItem.ExecutionOrder.Serial,
                interval_item=hourly,
            )
            sched = server.schedules.create(sched)

        server.schedules.add_to_schedule(schedule_id=sched.id, datasource=ds)
        server.auth.sign_out()
        return f"Attached '{datasource_name}' to schedule '{schedule_name}' (id={sched.id})."
    except Exception as e:
        return f"Schedule failed: {e}"

@tool
def tableau_job_status(job_id: str) -> str:
    """Check the status of a background job (refresh, publish, etc.)."""
    try:
        server = _server()
        job = server.jobs.get_by_id(job_id)
        server.auth.sign_out()
        return f"Job {job_id} status: {getattr(job, 'status', None) or getattr(job, 'finish_code', None)}"
    except Exception as e:
        return f"Could not query job {job_id}: {e}"
