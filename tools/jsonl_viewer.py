#!/usr/bin/env python3
"""Web UI viewer for JSONL files produced by alloylm eval runs.

Usage:
    python tools/jsonl_viewer.py work_dirs/debug/gsm8k_test.jsonl
    python tools/jsonl_viewer.py work_dirs/debug   # all JSONL files recursively
    python tools/jsonl_viewer.py a.jsonl results/  # files and folders can be mixed
    python tools/jsonl_viewer.py --plain a.jsonl   # print to stdout, no UI
    python tools/jsonl_viewer.py --port 9000 --host 0.0.0.0 a.jsonl
"""

import argparse
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import orjson

# The message keys the viewer already renders in their own blocks; every other
# key is shown as a JSON block of its own.
KNOWN_MESSAGE_KEYS = {"role", "content", "reasoning_content", "tool_calls", "tool_call_id"}


def resolve_jsonl_files(inputs: list[str]) -> list[str]:
    """Expand input files and directories into a stable, deduplicated file
    list."""
    files: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_dir():
            files.extend(sorted(candidate for candidate in path.rglob("*.jsonl") if candidate.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError(f"input does not exist: {value}")

    return list(dict.fromkeys(str(path.resolve()) for path in files))


def display_paths(files: list[str]) -> list[str]:
    """Return compact paths relative to the files' common directory."""
    common_path = Path(os.path.commonpath(files))
    root = common_path.parent if len(files) == 1 else common_path
    return [os.path.relpath(file, start=root) for file in files]


def record_search_text(record: dict) -> str:
    """Flatten a record into one searchable string."""
    parts = [str(record.get("id", ""))]
    for msg in record.get("messages") or []:
        parts.append(str(msg.get("role", "")))
        parts.append(str(msg.get("content", "")))
        parts.append(str(msg.get("reasoning_content", "")))
        parts.append(str(msg.get("tool_call_id", "")))
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or call
            parts.append(str(fn.get("name", "")))
            parts.append(str(fn.get("arguments", "")))
    parts.append(str(record.get("others", "")))
    return " ".join(parts)


def print_plain(files: list[str]) -> None:
    """Non-interactive fallback: pretty-print every record to stdout."""
    for path, name in zip(files, display_paths(files)):
        print(f"===== {name} =====")
        with open(path, "rb") as handle:
            index = -1
            for line in handle:
                if not line.strip():
                    continue
                index += 1
                record = orjson.loads(line)
                metric = record.get("metric", "-")
                finish_reason = record.get("finish_reason") or "-"
                tokens = " / ".join(
                    str(record.get(key, "-")) for key in ("input_tokens", "output_tokens", "total_tokens")
                )
                print(
                    f"[{index}] id: {record.get('id', '?')}  metric: {metric}  finish_reason: {finish_reason}  tokens: {tokens}"
                )
                for msg in record.get("messages") or []:
                    header = f"  ### {msg.get('role', '?')}"
                    call_id = msg.get("tool_call_id")
                    if call_id is not None:
                        header += f" (call {call_id})"
                    print(header)
                    reasoning = msg.get("reasoning_content")
                    if msg.get("role") == "assistant" and reasoning:
                        print(f"  reasoning:\n{reasoning}")
                    print(msg.get("content") or "")
                    if msg.get("role") != "assistant" and reasoning:
                        print(f"  reasoning:\n{reasoning}")
                    for call in msg.get("tool_calls") or []:
                        fn = call.get("function") or call
                        print(f"  - {fn.get('name', '?')} {fn.get('arguments') or ''}")
                    extra = {k: v for k, v in msg.items() if k not in KNOWN_MESSAGE_KEYS}
                    if extra:
                        print("  extra: " + orjson.dumps(extra, option=orjson.OPT_INDENT_2).decode())
                print()


PAGE_SIZE = 100


class JsonlStore:
    """One jsonl file: byte offsets now, parsed records on demand.

    ``ensure(n)`` walks the file until n offsets are known, so the first page
    costs a few hundred KB no matter how large the file is.  Offsets are kept
    for the whole session; a record's body is parsed only when it is opened and
    a handful of recent ones are cached for back-and-forth navigation.
    """

    def __init__(self, path: str, cache_size: int = 32) -> None:
        self.path = path
        self._handle = open(path, "rb")  # noqa: SIM115
        self._lock = threading.Lock()
        self._offsets: list[int] = []
        self._eof = False
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._cache_size = cache_size

    @property
    def known(self) -> int:
        return len(self._offsets)

    @property
    def complete(self) -> bool:
        return self._eof

    def ensure(self, count: int) -> None:
        """Index at least ``count`` records (or reach the end of the file)."""
        with self._lock:
            while len(self._offsets) < count and not self._eof:
                position = self._handle.tell()
                line = self._handle.readline()
                if not line:
                    self._eof = True
                    break
                if line.strip():
                    self._offsets.append(position)

    def get(self, index: int) -> dict:
        if index < 0:
            raise IndexError(index)
        self.ensure(index + 1)
        if index >= len(self._offsets):
            raise IndexError(index)
        with self._lock:
            cached = self._cache.get(index)
            if cached is not None:
                self._cache.move_to_end(index)
                return cached
            self._handle.seek(self._offsets[index])
            line = self._handle.readline()
        record = orjson.loads(line)
        with self._lock:
            self._cache[index] = record
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return record

    def search(self, query: str, limit: int = 200) -> list[int]:
        """Scan the file for a substring; stops after ``limit`` hits."""
        hits: list[int] = []
        index = -1
        with open(self.path, "rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                index += 1
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                if query in record_search_text(record).lower():
                    hits.append(index)
                    if len(hits) >= limit:
                        break
        return hits

    def close(self) -> None:
        self._handle.close()


def record_summary(index: int, record: dict) -> dict:
    """The few fields the list column needs; the body stays on disk."""
    messages = record.get("messages") or []
    return {
        "idx": index,
        "id": record.get("id", "?"),
        "metric": record.get("metric", -1.0),
        "finish_reason": str(record.get("finish_reason") or "-"),
        "input_tokens": record.get("input_tokens"),
        "output_tokens": record.get("output_tokens"),
        "num_steps": sum(1 for message in messages if message.get("role") == "assistant"),
    }


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>jsonl viewer</title>
<style>
:root{color-scheme:light;--bg:#fff;--panel:#f6f8fa;--border:#d0d7de;--fg:#1f2328;--dim:#636c76;--ok:#1a7f37;--bad:#cf222e;--accent:#0969da;--reason:#57606a;--tool:#9a6700}
*{box-sizing:border-box}
body{margin:0;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:12px;padding:9px 12px;border-bottom:1px solid var(--border);background:var(--panel)}
header strong{white-space:nowrap}
#search{flex:1;min-width:180px;background:var(--bg);border:1px solid var(--border);color:var(--fg);padding:6px 10px;border-radius:6px;outline:none}
#search:focus{border-color:var(--accent);box-shadow:0 0 0 2px #ddf4ff}
#matchinfo{color:var(--dim);white-space:nowrap}
main{flex:1;display:grid;grid-template-columns:minmax(210px,22vw) minmax(430px,38vw) minmax(480px,1fr);min-width:1120px;overflow:hidden}
.panel{min-width:0;overflow-y:auto;border-right:1px solid var(--border)}
.panel-title{position:sticky;top:0;z-index:2;margin:0;padding:8px 10px;border-bottom:1px solid var(--border);background:var(--panel);font:600 12px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:var(--dim);text-transform:uppercase;letter-spacing:.04em}
#tabs{padding:6px}
.tab{display:block;padding:7px 9px;cursor:pointer;border-radius:6px;color:var(--dim);overflow-wrap:anywhere}
.tab:hover{background:#f0f3f6;color:var(--fg)}
.tab.active{color:var(--fg);background:#ddf4ff;box-shadow:inset 3px 0 var(--accent)}
#list{overflow:auto}
#detail-panel{overflow-y:auto;border-right:0}
#detail{padding:14px 18px}
table{width:100%;border-collapse:collapse}
thead th{position:sticky;top:31px;background:var(--bg);text-align:left;color:var(--dim);font-weight:600;padding:5px 8px;border-bottom:1px solid var(--border);z-index:1}
tbody td{padding:5px 8px;border-bottom:1px solid #eaeef2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:430px}
tbody tr{cursor:pointer}
tbody tr:hover{background:#f6f8fa}
tbody tr.sel{background:#ddf4ff}
.metric{font-weight:700}.ok{color:var(--ok)}.bad{color:var(--bad)}.none{color:var(--dim)}
.meta{color:var(--dim);margin-bottom:14px}
.meta b{color:var(--fg)}
.msg{max-width:95%;margin:10px 0 18px;padding:10px 12px;border:1px solid var(--border);border-radius:10px;background:var(--panel)}
.msg.system{border-color:#b6d7f2;background:#f1f8ff}.msg.user{border-color:#b7dfbd;background:#f2fbf3}.msg.assistant{border-color:#e5cf86;background:#fffbea}.msg.tool{border-color:#d8c2eb;background:#fbf7ff}
.role{font-weight:700}
.role.system{color:#0969da}.role.user{color:#1a7f37}.role.assistant{color:var(--tool)}.role.tool{color:#8250df}
.callid{color:var(--dim);font-weight:400}
pre{white-space:pre-wrap;word-break:break-word;margin:6px 0}
.reasoning{background:#fff;border:1px solid #d8dee4;border-radius:6px;padding:6px 10px;margin:6px 0}
.rlbl{color:var(--reason);font-weight:700;margin-bottom:2px}
pre.toolcall{color:var(--tool)}
.others{color:var(--dim);margin-top:10px}
.extra{background:#f6f8fa;border:1px dashed #d0d7de;border-radius:6px;padding:6px 10px;margin:6px 0}
.extra .rlbl{color:var(--dim)}
button.showall{background:none;border:none;color:var(--accent);cursor:pointer;font:inherit;padding:0}
#footer{padding:6px 12px;border-top:1px solid var(--border);color:var(--dim);background:var(--panel)}
</style>
</head>
<body>
<header>
  <strong>JSONL viewer</strong>
  <input id="search" placeholder="search selected file  (id / messages / reasoning / tool calls / others)" spellcheck="false">
  <span id="matchinfo"></span>
</header>
<main>
  <aside id="files" class="panel"><h2 class="panel-title">Files</h2><div id="tabs"></div></aside>
  <section id="list" class="panel"><h2 class="panel-title">Items</h2><table><thead><tr><th>#</th><th>metric</th><th>finish</th><th>tokens</th><th>steps</th><th>id</th></tr></thead><tbody id="rows"></tbody></table></section>
  <section id="detail-panel" class="panel"><h2 class="panel-title">Content</h2><div id="detail"><div class="meta">Select a record.</div></div></section>
</main>
<div id="footer">j/k or &#8593;/&#8595; move &middot; Enter open &middot; g/G top/bottom &middot; / focus search &middot; n/N matches &middot; click row to inspect</div>
<script>
const $ = (s) => document.querySelector(s);
const rowsEl = $('#rows'), detailEl = $('#detail'), searchEl = $('#search'), tabsEl = $('#tabs'), matchEl = $('#matchinfo');
let files = [], fileIdx = 0, loaded = [], known = 0, complete = false, loading = false,
    cursor = -1, matches = [], matchIdx = -1;
const PAGE = 100;
let fullTexts = new Map();

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

function escTrunc(s, limit = 4000) {
  s = String(s);
  if (s.length <= limit) return esc(s);
  const id = 'ft' + fullTexts.size;
  const headLength = Math.ceil(limit / 2);
  const tailLength = Math.floor(limit / 2);
  fullTexts.set(id, s);
  return `<span class="truncated">${esc(s.slice(0, headLength))}\n&#8230; <button class="showall" data-id="${id}">show all (${s.length} chars)</button> &#8230;\n${esc(s.slice(-tailLength))}</span>`;
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(res.status + ' ' + path);
  return res.json();
}

async function loadFiles() {
  files = await api('/api/files');
  renderTabs();
  await selectFile(0);
}

function renderTabs() {
  tabsEl.innerHTML = files.map((f, i) => {
    const count = i === fileIdx ? (complete ? loaded.length : loaded.length + '+') : (f.total ?? '?');
    return `<span class="tab${i === fileIdx ? ' active' : ''}" data-i="${i}" title="${esc(f.name)}">${esc(f.name)} <span style="opacity:.6">(${count})</span></span>`;
  }).join('');
}

async function selectFile(i) {
  fileIdx = i; cursor = -1; matches = []; matchIdx = -1; matchEl.textContent = '';
  loaded = []; known = 0; complete = false; loading = false;
  detailEl.innerHTML = '<div class="meta">Select a record.</div>';
  await loadMore();
  renderTabs(); renderList();
}

async function loadMore() {
  if (loading || complete) return;
  loading = true;
  const start = loaded.length;
  try {
    const page = await api(`/api/list/${fileIdx}?start=${start}&limit=${PAGE}`);
    loaded = loaded.concat(page.items);
    known = page.known;
    complete = page.complete;
  } finally {
    loading = false;
  }
  renderTabs();
  renderList();
}

async function ensureLoadedFor(i) {
  while (!complete && loaded.length <= i) await loadMore();
}

function metricCell(m) {
  if (m === 1.0) return '<td class="metric ok">ok</td>';
  if (m === 0.0) return '<td class="metric bad">bad</td>';
  return '<td class="metric none">-</td>';
}

function renderList() {
  const rows = loaded.map((r) =>
    `<tr data-i="${r.idx}" class="${r.idx === cursor ? 'sel' : ''}"><td>${r.idx}</td>${metricCell(r.metric)}<td>${esc(r.finish_reason)}</td><td>${r.input_tokens ?? '-'}/${r.output_tokens ?? '-'}</td><td>${r.num_steps}</td><td>${esc(r.id)}</td></tr>`
  ).join('');
  const tail = complete ? '' : `<tr><td colspan="6" class="meta" id="loading-row">loading... ${loaded.length} loaded</td></tr>`;
  rowsEl.innerHTML = rows + tail;
  const sel = rowsEl.querySelector('tr.sel');
  if (sel) sel.scrollIntoView({ block: 'nearest' });
}


async function openRecord(i) {
  cursor = i;
  await ensureLoadedFor(i);
  renderList();
  const rec = await api(`/api/record/${fileIdx}/${i}`);
  renderDetail(rec);
}

const KNOWN_MESSAGE_KEYS = new Set(['role', 'content', 'reasoning_content', 'tool_calls', 'tool_call_id']);

function metricLabel(m) { return m === 1.0 ? 'ok' : (m === 0.0 ? 'bad' : '-'); }

function renderDetail(rec) {
  fullTexts.clear();
  const infer = rec.infer_args || {};
  const sample = infer.sample_args ? esc(JSON.stringify(infer.sample_args)) : '-';
  let html = `<div class="meta"><b>#${cursor}</b> &nbsp; id: <b>${esc(rec.id ?? '-')}</b><br>` +
    `model: ${esc(infer.model_name || '-')} &middot; metric: ${metricLabel(rec.metric)} &middot; finish_reason: ${esc(rec.finish_reason || '-')}<br>` +
    `tokens: ${rec.input_tokens ?? '-'} / ${rec.output_tokens ?? '-'} / ${rec.total_tokens ?? '-'} &middot; sample_args: ${sample}</div>`;
  for (const m of (rec.messages || [])) html += msgHtml(m);
  const others = rec.others;
  if (others && typeof others === 'object') {
    html += '<div class="others">### others</div>';
    for (const [k, v] of Object.entries(others)) {
      const vs = (v && typeof v === 'object') ? '{' + Object.keys(v).slice(0, 8).join(', ') + '}' : String(v);
      html += `<div class="others">${esc(k)}: ${escTrunc(vs, 500)}</div>`;
    }
  }
  detailEl.innerHTML = html;
  detailEl.parentElement.scrollTop = 0;
}

function msgHtml(m) {
  const role = m.role || '?';
  const roleClass = ['system', 'user', 'assistant', 'tool'].includes(role) ? role : '';
  let head = `<div class="role ${roleClass}">### ${esc(role)}` +
    (m.tool_call_id ? ` <span class="callid">(call ${esc(m.tool_call_id)})</span>` : '') + '</div>';
  const reasoning = m.reasoning_content
    ? `<div class="reasoning"><div class="rlbl">reasoning</div><pre>${escTrunc(m.reasoning_content)}</pre></div>`
    : '';
  const content = m.content ? `<pre>${escTrunc(m.content)}</pre>` : '';
  const tools = (m.tool_calls || []).map((c) => {
    const f = c.function || c;
    const args = typeof f.arguments === 'string' ? f.arguments : JSON.stringify(f.arguments || '');
    return `<pre class="toolcall">- ${esc(f.name || '?')} ${esc(args)}</pre>`;
  }).join('');
  const extraKeys = Object.keys(m).filter((k) => !KNOWN_MESSAGE_KEYS.has(k));
  const extras = extraKeys.map((k) => {
    const value = m[k];
    const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    return `<div class="extra"><div class="rlbl">${esc(k)}</div><pre>${escTrunc(text)}</pre></div>`;
  }).join('');
  const body = role === 'assistant' ? reasoning + content + tools + extras : content + reasoning + tools + extras;
  return `<div class="msg ${roleClass}">${head}${body}</div>`;
}

tabsEl.addEventListener('click', (e) => {
  const tab = e.target.closest('.tab');
  if (tab) selectFile(Number(tab.dataset.i));
});
rowsEl.addEventListener('click', (e) => {
  const tr = e.target.closest('tr');
  if (tr) openRecord(Number(tr.dataset.i));
});
$('#list').addEventListener('scroll', (event) => {
  const el = event.currentTarget;
  if (el.scrollTop + el.clientHeight >= el.scrollHeight - 400) loadMore();
});
detailEl.addEventListener('click', (e) => {
  if (!e.target.classList.contains('showall')) return;
  const full = fullTexts.get(e.target.dataset.id);
  e.target.closest('.truncated').outerHTML = `<pre>${esc(full)}</pre>`;
});

let searchTimer = null;
searchEl.addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(runSearch, 300);
});
async function runSearch() {
  const q = searchEl.value.trim();
  if (!q) { matches = []; matchIdx = -1; matchEl.textContent = ''; return; }
  matches = await api(`/api/search/${fileIdx}?q=${encodeURIComponent(q)}`);
  matchIdx = -1;
  matchEl.textContent = matches.length ? `${matches.length} match(es)` : 'no matches';
  if (matches.length) jumpMatch(1);
}
function jumpMatch(step) {
  if (!matches.length) return;
  matchIdx = (matchIdx + step + matches.length) % matches.length;
  openRecord(matches[matchIdx]);
}

document.addEventListener('keydown', (e) => {
  if (e.target === searchEl) {
    if (e.key === 'Enter') { e.preventDefault(); runSearch(); }
    if (e.key === 'Escape') { searchEl.value = ''; matches = []; matchIdx = -1; matchEl.textContent = ''; }
    return;
  }
  if (e.key === '/') { e.preventDefault(); searchEl.focus(); }
  else if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); cursor = Math.min(cursor + 1, loaded.length - 1); renderList(); }
  else if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); cursor = Math.max(0, cursor - 1); renderList(); }
  else if (e.key === 'Enter') { if (cursor >= 0) openRecord(cursor); }
  else if (e.key === 'g') { cursor = 0; renderList(); }
  else if (e.key === 'G') { cursor = loaded.length - 1; renderList(); }
  else if (e.key === 'n') { jumpMatch(1); }
  else if (e.key === 'N') { jumpMatch(-1); }
});

loadFiles();
</script>
</body>
</html>
"""


def run_web(files: list[str], host: str, port: int) -> None:
    """Serve the lazily indexed files on http://host:port."""
    import webbrowser

    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    file_names = display_paths(files)
    stores = [JsonlStore(path) for path in files]

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/api/files")
    def api_files():
        return [
            {"name": name, "total": store.known if store.complete else None} for name, store in zip(file_names, stores)
        ]

    @app.get("/api/list/{file_idx}")
    def api_list(file_idx: int, start: int = 0, limit: int = PAGE_SIZE):
        if not 0 <= file_idx < len(stores):
            raise HTTPException(status_code=404, detail="file index out of range")
        store = stores[file_idx]
        start = max(0, start)
        limit = max(1, min(limit, 1000))
        store.ensure(start + limit)
        end = min(start + limit, store.known)
        return {
            "start": start,
            "items": [record_summary(index, store.get(index)) for index in range(start, end)],
            "known": store.known,
            "complete": store.complete,
        }

    @app.get("/api/record/{file_idx}/{idx}")
    def api_record(file_idx: int, idx: int):
        if not 0 <= file_idx < len(stores):
            raise HTTPException(status_code=404, detail="file index out of range")
        try:
            return stores[file_idx].get(idx)
        except IndexError:
            raise HTTPException(status_code=404, detail="record index out of range")

    @app.get("/api/search/{file_idx}")
    def api_search(file_idx: int, q: str = "", limit: int = 200):
        if not 0 <= file_idx < len(stores):
            raise HTTPException(status_code=404, detail="file index out of range")
        query = q.lower()
        if not query:
            return []
        return stores[file_idx].search(query, limit=limit)

    url = f"http://{host}:{port}"
    print(f"jsonl_viewer: serving {len(files)} file(s) at {url}")
    try:
        webbrowser.open(url)
    except (OSError, webbrowser.Error):
        print("jsonl_viewer: could not open a browser; visit the URL manually")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="JSONL file(s) or directories to view")
    parser.add_argument("--plain", action="store_true", help="print records to stdout instead of launching a UI")
    parser.add_argument("--host", default="127.0.0.1", help="web UI bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="web UI port (default 8765)")
    args = parser.parse_args()

    try:
        files = resolve_jsonl_files(args.inputs)
    except ValueError as error:
        parser.error(str(error))
    if not files:
        parser.error("no JSONL files found in the provided inputs")

    if args.plain:
        print_plain(files)
        return 0

    run_web(files, args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
