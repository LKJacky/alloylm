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
from functools import lru_cache
from pathlib import Path

from alloylm.utils import load_jsonl


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
        for index, record in enumerate(load_jsonl(path)):
            metric = record.get("metric", "-")
            finish_reason = record.get("finish_reason") or "-"
            tokens = " / ".join(str(record.get(k, "-")) for k in ("input_tokens", "output_tokens", "total_tokens"))
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
            print()


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
let files = [], fileIdx = 0, records = [], cursor = -1, matches = [], matchIdx = -1;
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
  tabsEl.innerHTML = files.map((f, i) =>
    `<span class="tab${i === fileIdx ? ' active' : ''}" data-i="${i}" title="${esc(f.name)}">${esc(f.name)} <span style="opacity:.6">(${f.count})</span></span>`
  ).join('');
}

async function selectFile(i) {
  fileIdx = i; cursor = -1; matches = []; matchIdx = -1; matchEl.textContent = '';
  records = await api(`/api/list/${i}`);
  renderTabs(); renderList();
  detailEl.innerHTML = '<div class="meta">Select a record.</div>';
}

function metricCell(m) {
  if (m === 1.0) return '<td class="metric ok">ok</td>';
  if (m === 0.0) return '<td class="metric bad">bad</td>';
  return '<td class="metric none">-</td>';
}

function renderList() {
  rowsEl.innerHTML = records.map((r) =>
    `<tr data-i="${r.idx}" class="${r.idx === cursor ? 'sel' : ''}"><td>${r.idx}</td>${metricCell(r.metric)}<td>${esc(r.finish_reason)}</td><td>${r.input_tokens ?? '-'}/${r.output_tokens ?? '-'}</td><td>${r.num_steps}</td><td>${esc(r.id)}</td></tr>`
  ).join('');
  const sel = rowsEl.querySelector('tr.sel');
  if (sel) sel.scrollIntoView({ block: 'nearest' });
}

async function openRecord(i) {
  cursor = i; renderList();
  const rec = await api(`/api/record/${fileIdx}/${i}`);
  renderDetail(rec);
}

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
  const body = role === 'assistant' ? reasoning + content + tools : content + reasoning + tools;
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
  else if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); cursor = Math.min(cursor + 1, records.length - 1); renderList(); }
  else if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); cursor = Math.max(0, cursor - 1); renderList(); }
  else if (e.key === 'Enter') { if (cursor >= 0) openRecord(cursor); }
  else if (e.key === 'g') { cursor = 0; renderList(); }
  else if (e.key === 'G') { cursor = records.length - 1; renderList(); }
  else if (e.key === 'n') { jumpMatch(1); }
  else if (e.key === 'N') { jumpMatch(-1); }
});

loadFiles();
</script>
</body>
</html>
"""


def run_web(files: list[str], host: str, port: int) -> None:
    """Serve a web UI for the given JSONL files on http://host:port."""
    import webbrowser

    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    file_names = display_paths(files)
    counts = []
    for path in files:
        with open(path, "rb") as file:
            counts.append(sum(bool(line.strip()) for line in file))

    @lru_cache(maxsize=2)
    def get_records(file_idx: int) -> list[dict]:
        return load_jsonl(files[file_idx])

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/api/files")
    def api_files():
        return [{"name": name, "count": count} for name, count in zip(file_names, counts)]

    @app.get("/api/list/{file_idx}")
    def api_list(file_idx: int):
        if not 0 <= file_idx < len(files):
            raise HTTPException(status_code=404, detail="file index out of range")
        records = get_records(file_idx)
        return [
            {
                "idx": i,
                "id": rec.get("id", "?"),
                "metric": rec.get("metric", -1.0),
                "finish_reason": str(rec.get("finish_reason") or "-"),
                "input_tokens": rec.get("input_tokens"),
                "output_tokens": rec.get("output_tokens"),
                "num_steps": sum(msg.get("role") == "assistant" for msg in rec.get("messages") or []),
            }
            for i, rec in enumerate(records)
        ]

    @app.get("/api/record/{file_idx}/{idx}")
    def api_record(file_idx: int, idx: int):
        if not 0 <= file_idx < len(files):
            raise HTTPException(status_code=404, detail="file index out of range")
        records = get_records(file_idx)
        if not 0 <= idx < len(records):
            raise HTTPException(status_code=404, detail="record index out of range")
        return records[idx]

    @app.get("/api/search/{file_idx}")
    def api_search(file_idx: int, q: str = ""):
        if not 0 <= file_idx < len(files):
            raise HTTPException(status_code=404, detail="file index out of range")
        query = q.lower()
        if not query:
            return []
        return [i for i, record in enumerate(get_records(file_idx)) if query in record_search_text(record).lower()]

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
