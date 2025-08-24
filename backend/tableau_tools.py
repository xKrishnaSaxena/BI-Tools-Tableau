# backend/tableau_tools.py
import os, time, uuid, json, io, csv, base64
from typing import Optional, Tuple, Dict, Any, List

import requests
import jwt
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

TABLEAU_SERVER_URL = (os.getenv("TABLEAU_SERVER_URL", "") or "").rstrip("/")
TABLEAU_CLIENT_ID = os.getenv("TABLEAU_CLIENT_ID", "")
TABLEAU_SECRET_ID = os.getenv("TABLEAU_SECRET_ID", "")
TABLEAU_SECRET_VALUE = os.getenv("TABLEAU_SECRET_VALUE", "")
TABLEAU_USERNAME = os.getenv("TABLEAU_USERNAME", "")
TABLEAU_SITE_CONTENT_URL = os.getenv("TABLEAU_SITE_CONTENT_URL", "")  # "" for Default
TABLEAU_API_VERSION = os.getenv("TABLEAU_API_VERSION", "")            # optional

# cached auth
_AUTH_TOKEN: Optional[str] = None
_SITE_ID: Optional[str] = None
_TOKEN_EXPIRES_AT: Optional[int] = None  # epoch seconds


def _require(name: str, value: str, allow_empty: bool = False) -> str:
    if value is None:
        value = ""
    if not allow_empty and not str(value).strip():
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _get_api_version() -> str:
    if TABLEAU_API_VERSION:
        return TABLEAU_API_VERSION
    url = f"{TABLEAU_SERVER_URL}/api/3.0/serverinfo"
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        r.raise_for_status()
        return r.json()["serverInfo"]["restApiVersion"]
    except Exception:
        # Safe modern fallback; adjust if your server is older
        return "3.26"


def _make_jwt(ttl_sec: int = 300) -> str:
    # Scopes for this MVP:
    # - content:read          -> list sites/workbooks/views
    # - views:download        -> Query View Data/Image
    # - tasks:run             -> refresh datasources/workbooks
    # - jobs:read             -> query job status
    payload = {
        "iss": _require("TABLEAU_CLIENT_ID", TABLEAU_CLIENT_ID),
        "sub": _require("TABLEAU_USERNAME", TABLEAU_USERNAME),
        "aud": "tableau",
        "exp": int(time.time()) + ttl_sec,
        "jti": str(uuid.uuid4()),
        "scp": [
            "tableau:content:read",
            "tableau:views:download",
            "tableau:tasks:run",
            "tableau:jobs:read",
        ],
    }
    headers = {"kid": _require("TABLEAU_SECRET_ID", TABLEAU_SECRET_ID), "alg": "HS256"}
    return jwt.encode(payload, _require("TABLEAU_SECRET_VALUE", TABLEAU_SECRET_VALUE),
                      algorithm="HS256", headers=headers)


def get_tableau_api_token(force: bool = False) -> Tuple[str, str]:
    """Sign in with Connected Apps JWT and cache token + site id."""
    global _AUTH_TOKEN, _SITE_ID, _TOKEN_EXPIRES_AT
    if (not force and _AUTH_TOKEN and _SITE_ID and _TOKEN_EXPIRES_AT
            and time.time() < _TOKEN_EXPIRES_AT - 60):
        return _AUTH_TOKEN, _SITE_ID

    _require("TABLEAU_SERVER_URL", TABLEAU_SERVER_URL)
    api_ver = _get_api_version()
    url = f"{TABLEAU_SERVER_URL}/api/{api_ver}/auth/signin"
    jwt_token = _make_jwt()

    r = requests.post(
        url,
        json={"credentials": {"jwt": jwt_token, "site": {"contentUrl": TABLEAU_SITE_CONTENT_URL or ""}}},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        allow_redirects=False,
        timeout=30
    )

    if 300 <= r.status_code < 400:
        loc = r.headers.get("Location", "")
        raise RuntimeError(f"Signin returned redirect {r.status_code} to {loc}. Check TABLEAU_SERVER_URL.")

    r.raise_for_status()
    data = r.json()
    _AUTH_TOKEN = data["credentials"]["token"]
    _SITE_ID = data["credentials"]["site"]["id"]
    _TOKEN_EXPIRES_AT = int(time.time()) + 5 * 60
    return _AUTH_TOKEN, _SITE_ID


def _auth_headers(token: str) -> Dict[str, str]:
    return {"X-Tableau-Auth": token, "Accept": "application/json"}


# ---------- Helpers to find content ----------
def _query_all(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code in (401, 403):
        token, _ = get_tableau_api_token(force=True)
        headers = _auth_headers(token)
        r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def _find_workbook_id(workbook_name: str) -> Optional[str]:
    token, site_id = get_tableau_api_token()
    api = _get_api_version()
    headers = _auth_headers(token)
    # filter by name for speed
    url = (f"{TABLEAU_SERVER_URL}/api/{api}/sites/{site_id}/workbooks"
           f"?filter=name:eq:{requests.utils.quote(workbook_name)}&pageSize=1000")
    data = _query_all(url, headers)
    for wb in data.get("workbooks", {}).get("workbook", []):
        if wb.get("name", "").lower() == workbook_name.lower():
            return wb.get("id")
    return None


def _find_view(view_name: str, workbook_name: Optional[str]) -> Optional[Dict[str, str]]:
    token, site_id = get_tableau_api_token()
    api = _get_api_version()
    headers = _auth_headers(token)

    if workbook_name:
        wb_id = _find_workbook_id(workbook_name)
        if not wb_id:
            return None
        url = f"{TABLEAU_SERVER_URL}/api/{api}/sites/{site_id}/workbooks/{wb_id}/views?pageSize=1000"
        data = _query_all(url, headers)
        for v in data.get("views", {}).get("view", []):
            if v.get("name", "").lower() == view_name.lower():
                return {"id": v.get("id"), "name": v.get("name")}
        return None

    # search entire site by filter name
    url = (f"{TABLEAU_SERVER_URL}/api/{api}/sites/{site_id}/views"
           f"?filter=name:eq:{requests.utils.quote(view_name)}&pageSize=1000")
    data = _query_all(url, headers)
    views = data.get("views", {}).get("view", [])
    if views:
        v = views[0]
        return {"id": v.get("id"), "name": v.get("name")}
    return None


def _vf_params(filters: Optional[Dict[str, Any]]) -> Dict[str, str]:
    # convert {"Region":"APAC","Year":"2024"} -> {"vf_Region":"APAC","vf_Year":"2024"}
    if not filters:
        return {}
    out = {}
    for k, v in filters.items():
        out[f"vf_{k}"] = str(v)
    return out


# ---------- LangChain Tools ----------
@tool
def list_tableau_projects() -> str:
    """List all Tableau projects you have access to."""
    try:
        token, site_id = get_tableau_api_token()
        api = _get_api_version()
        headers = _auth_headers(token)
        url = f"{TABLEAU_SERVER_URL}/api/{api}/sites/{site_id}/projects?pageSize=1000"
        data = _query_all(url, headers)
        projects = data.get("projects", {}).get("project", [])
        if not projects:
            return "No projects found on the Tableau site."
        names = ", ".join(p.get("name", "Unnamed") for p in projects)
        return f"Projects: {names}"
    except Exception as e:
        return f"An error occurred while listing projects: {e}"


@tool
def list_tableau_workbooks() -> str:
    """List workbooks that are visible to you."""
    try:
        token, site_id = get_tableau_api_token()
        api = _get_api_version()
        headers = _auth_headers(token)
        url = f"{TABLEAU_SERVER_URL}/api/{api}/sites/{site_id}/workbooks?pageSize=1000"
        data = _query_all(url, headers)
        wbs = data.get("workbooks", {}).get("workbook", [])
        if not wbs:
            return "No workbooks found."
        names = ", ".join(w.get("name", "Unnamed") for w in wbs)
        return f"Workbooks: {names}"
    except Exception as e:
        return f"Error listing workbooks: {e}"


@tool
def list_tableau_views(workbook_name: str = "") -> str:
    """List views. Optionally pass a workbook_name to narrow the list."""
    try:
        token, site_id = get_tableau_api_token()
        api = _get_api_version()
        headers = _auth_headers(token)
        if workbook_name:
            wb_id = _find_workbook_id(workbook_name)
            if not wb_id:
                return f"Workbook '{workbook_name}' not found."
            url = f"{TABLEAU_SERVER_URL}/api/{api}/sites/{site_id}/workbooks/{wb_id}/views?pageSize=1000"
        else:
            url = f"{TABLEAU_SERVER_URL}/api/{api}/sites/{site_id}/views?pageSize=1000"
        data = _query_all(url, headers)
        views = data.get("views", {}).get("view", [])
        if not views:
            return "No views found."
        names = ", ".join(v.get("name", "Unnamed") for v in views)
        if workbook_name:
            return f"Views in '{workbook_name}': {names}"
        return f"Views: {names}"
    except Exception as e:
        return f"Error listing views: {e}"


@tool
def refresh_tableau_datasource(datasource_name: str) -> str:
    """
    Run an extract refresh on a published data source.
    Requires scopes: tableau:content:read, tableau:tasks:run.
    """
    try:
        token, site_id = get_tableau_api_token()
        api = _get_api_version()
        headers = _auth_headers(token)
        headers["Content-Type"] = "application/json"

        # find datasource id
        url = f"{TABLEAU_SERVER_URL}/api/{api}/sites/{site_id}/datasources?pageSize=1000"
        data = _query_all(url, headers)
        dss = data.get("datasources", {}).get("datasource", [])
        ds_id = next((d.get("id") for d in dss if d.get("name", "").lower() == datasource_name.lower()), None)
        if not ds_id:
            return f"Error: Data source '{datasource_name}' not found."

        # run refresh
        refresh_url = f"{TABLEAU_SERVER_URL}/api/{api}/sites/{site_id}/datasources/{ds_id}/refresh"
        r = requests.post(refresh_url, headers=headers, json={}, timeout=30)
        if r.status_code in (401, 403):
            token, site_id = get_tableau_api_token(force=True)
            headers["X-Tableau-Auth"] = token
            r = requests.post(refresh_url, headers=headers, json={}, timeout=30)

        try:
            r.raise_for_status()
        except requests.HTTPError:
            return f"Tableau error {r.status_code} while refreshing: {(r.text or '')[:500]}"

        job_id = (r.json().get("job", {}) or {}).get("id")
        return f"Started refresh for '{datasource_name}'. Job ID: {job_id}."
    except Exception as e:
        return f"An error occurred during refresh: {e}"


@tool
def tableau_job_status(job_id: str) -> str:
    """Query an async job status (requires tableau:jobs:read and API 3.24+)."""
    try:
        token, site_id = get_tableau_api_token()
        api = _get_api_version()
        headers = _auth_headers(token)
        url = f"{TABLEAU_SERVER_URL}/api/{api}/sites/{site_id}/jobs/{job_id}"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code in (401, 403):
            token, site_id = get_tableau_api_token(force=True)
            headers = _auth_headers(token)
            r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        state = r.json().get("job", {}).get("status")
        return f"Job {job_id} status: {state}"
    except Exception as e:
        return f"Could not query job {job_id}: {e}"


@tool
def tableau_get_view_image(view_name: str, workbook_name: str = "", filters_json: str = "") -> str:
    """
    Return a PNG image (data URL) for a view. Optional filters_json like '{"Region":"APAC"}'.
    Requires scope: tableau:views:download.
    """
    try:
        filt = json.loads(filters_json) if filters_json else {}
        v = _find_view(view_name, workbook_name or None)
        if not v:
            return json.dumps({"text": f"View '{view_name}' not found.", "image": None})
        token, site_id = get_tableau_api_token()
        api = _get_api_version()
        headers = {"X-Tableau-Auth": token}
        params = {"maxAge": "60"}
        params.update(_vf_params(filt))
        url = f"{TABLEAU_SERVER_URL}/api/{api}/sites/{site_id}/views/{v['id']}/image"
        r = requests.get(url, headers=headers, params=params, timeout=60)
        if r.status_code in (401, 403):
            token, site_id = get_tableau_api_token(force=True)
            headers["X-Tableau-Auth"] = token
            r = requests.get(url, headers=headers, params=params, timeout=60)
        r.raise_for_status()
        img_b64 = base64.b64encode(r.content).decode("utf-8")
        return json.dumps({"text": f"Rendered view '{v['name']}'.", "image": f"data:image/png;base64,{img_b64}"})
    except Exception as e:
        return json.dumps({"text": f"Image fetch failed: {e}", "image": None})


@tool
def tableau_get_view_data(view_name: str, workbook_name: str = "", filters_json: str = "", max_rows: int = 200) -> str:
    """
    Download a view's data as CSV and return as JSON (up to max_rows).
    Requires scope: tableau:views:download.
    """
    try:
        filt = json.loads(filters_json) if filters_json else {}
        v = _find_view(view_name, workbook_name or None)
        if not v:
            return json.dumps({"text": f"View '{view_name}' not found.", "table": []})
        token, site_id = get_tableau_api_token()
        api = _get_api_version()
        headers = {"X-Tableau-Auth": token, "Accept": "text/csv"}
        params = {"maxAge": "60"}
        params.update(_vf_params(filt))
        url = f"{TABLEAU_SERVER_URL}/api/{api}/sites/{site_id}/views/{v['id']}/data"
        r = requests.get(url, headers=headers, params=params, timeout=60)
        if r.status_code in (401, 403):
            token, site_id = get_tableau_api_token(force=True)
            headers["X-Tableau-Auth"] = token
            r = requests.get(url, headers=headers, params=params, timeout=60)
        r.raise_for_status()
        text = r.text
        f = io.StringIO(text)
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)
        return json.dumps({"text": f"Returned {len(rows)} rows from '{v['name']}'.", "table": rows})
    except Exception as e:
        return json.dumps({"text": f"Data fetch failed: {e}", "table": []})

@tool
def list_tableau_datasources() -> str:
    """List published datasources (helps you find the exact name for refresh)."""
    try:
        token, site_id = get_tableau_api_token()
        api = _get_api_version()
        headers = _auth_headers(token)
        url = f"{TABLEAU_SERVER_URL}/api/{api}/sites/{site_id}/datasources?pageSize=1000"
        data = _query_all(url, headers)
        dss = data.get("datasources", {}).get("datasource", [])
        if not dss:
            return "No datasources found."
        names = ", ".join(d.get("name", "Unnamed") for d in dss)
        return f"Datasources: {names}"
    except Exception as e:
        return f"Error listing datasources: {e}"
