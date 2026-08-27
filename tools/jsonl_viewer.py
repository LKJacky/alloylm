#!/usr/bin/env python3
"""Interactive TUI viewer for JSONL files produced by alloylm eval runs.

Usage:
    python tools/jsonl_viewer.py work_dirs/debug/gsm8k_test.jsonl
    python tools/jsonl_viewer.py a.jsonl b.jsonl   # Tab switches between files
    python tools/jsonl_viewer.py --plain a.jsonl   # print to stdout, no TUI
"""

import argparse
import curses
import sys
import unicodedata

from alloylm.utils import load_jsonl

ESC = 27

ROLE_COLORS = {"system": 4, "user": 5, "assistant": 6, "tool": 7}

HELP_TEXT = [
    "jsonl_viewer keys",
    "",
    "j/k or up/down   move cursor",
    "g/G or home/end  jump to first/last record",
    "Enter            inspect record (messages etc.)",
    "/                search (match in id/messages/others)",
    "n/N              next/previous search match",
    "Tab              switch to next file",
    "h                this help",
    "q                quit",
]

LIST_FOOTER = "j/k move  g/G top/bottom  Enter inspect  / search  n/N match  Tab file  h help  q quit"
DETAIL_FOOTER = "j/k scroll  g/G top/bottom  q/Enter back"


def display_width(text: str) -> int:
    """Return the number of terminal columns `text` occupies."""
    width = 0
    for ch in text:
        if ch == "\t":
            width += 4 - (width % 4)
        elif unicodedata.category(ch) != "Cc":
            width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def truncate(text: str, width: int) -> str:
    """Truncate `text` so it fits into at most `width` terminal columns."""
    if width <= 0 or display_width(text) <= width:
        return text
    out = []
    used = 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if used + cw > width - 1:
            out.append("…")
            return "".join(out)
        out.append(ch)
        used += cw
    return "".join(out)


def wrap_text(text: str, width: int) -> list[str]:
    """Hard-wrap `text` at `width` terminal columns, preserving newlines."""
    if width <= 0:
        return []
    lines = []
    for raw_line in text.split("\n"):
        buf = []
        used = 0
        for ch in raw_line:
            cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if used + cw > width:
                lines.append("".join(buf))
                buf = [ch]
                used = cw
            else:
                buf.append(ch)
                used += cw
        lines.append("".join(buf))
    return lines


def summarize(record: dict, index: int, width: int) -> tuple[str, int]:
    """Build the one-line list entry for `record` plus its color attr."""
    metric = record.get("metric", -1.0)
    if metric == 1.0:
        metric_s, attr = "ok ", 2
    elif metric == 0.0:
        metric_s, attr = "bad", 3
    else:
        metric_s, attr = "  -", 0

    finish_reason = str(record.get("finish_reason") or "-")
    in_t = record.get("input_tokens")
    out_t = record.get("output_tokens")
    tokens = f"{in_t}/{out_t}" if isinstance(in_t, int) and isinstance(out_t, int) else "-/-"

    text = f"{index:>6}  {metric_s}  {finish_reason:<9} {tokens:>9}  {record.get('id', '?')}"
    return truncate(text, width), attr


def record_search_text(record: dict) -> str:
    """Flatten a record into one searchable string."""
    parts = [str(record.get("id", ""))]
    for msg in record.get("messages") or []:
        parts.append(str(msg.get("role", "")))
        parts.append(str(msg.get("content", "")))
    parts.append(str(record.get("others", "")))
    return " ".join(parts)


def detail_lines(record: dict, width: int) -> list[tuple[str, int]]:
    """Render one record as (text, color-attr) lines for the detail view."""
    lines: list[tuple[str, int]] = []
    lines.append((truncate(f"id: {record.get('id', '-')}", width), 1))

    infer = record.get("infer_args") or {}
    model = infer.get("model_name", "-")
    metric = record.get("metric", -1.0)
    finish_reason = record.get("finish_reason") or "-"
    tokens = " / ".join(str(record.get(k, "-")) for k in ("input_tokens", "output_tokens", "total_tokens"))
    lines.append(
        (truncate(f"model: {model}  metric: {metric}  finish_reason: {finish_reason}  tokens: {tokens}", width), 0)
    )

    sample_args = infer.get("sample_args")
    if isinstance(sample_args, dict) and sample_args:
        lines.append((truncate(f"sample_args: {sample_args}", width), 0))
    lines.append(("", 0))

    for msg in record.get("messages") or []:
        role = str(msg.get("role", "?"))
        attr = ROLE_COLORS.get(role, 0)
        lines.append((truncate(f"### {role}", width), attr | curses.A_BOLD))
        for line in wrap_text(str(msg.get("content") or ""), width):
            lines.append((line, attr))
        lines.append(("", 0))

    others = record.get("others")
    if isinstance(others, dict) and others:
        lines.append(("### others", 1 | curses.A_BOLD))
        for key, value in others.items():
            if isinstance(value, dict):
                value_s = "{" + ", ".join(str(k) for k in list(value)[:8]) + "}"
            else:
                value_s = str(value)
            lines.append((truncate(f"{key}: {value_s}", width), 0))
    return lines


class App:
    def __init__(self, stdscr, files: list[str]):
        self.stdscr = stdscr
        self.files = files
        self.records = [load_jsonl(f) for f in files]
        self.file_idx = 0
        self.cursor = 0
        self.top = 0
        self.query = ""
        self.matches: list[int] = []
        self.match_idx = -1
        self._search_cache: list[list[str | None]] = [[] for _ in files]

        curses.curs_set(0)
        try:
            curses.use_default_colors()
        except curses.error:
            pass
        if curses.has_colors():
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.init_pair(4, curses.COLOR_CYAN, -1)
            curses.init_pair(5, curses.COLOR_GREEN, -1)
            curses.init_pair(6, curses.COLOR_YELLOW, -1)
            curses.init_pair(7, curses.COLOR_MAGENTA, -1)
            curses.init_pair(8, curses.COLOR_BLUE, -1)
        self.colors = {
            "header": curses.color_pair(1),
            "ok": curses.color_pair(2),
            "bad": curses.color_pair(3),
            "match": curses.color_pair(8),
        }

    # -- drawing helpers ---------------------------------------------------

    def addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        try:
            self.stdscr.addstr(y, x, text, attr)
        except curses.error:
            pass

    def draw_header(self, text: str) -> None:
        _, width = self.stdscr.getmaxyx()
        self.addstr(0, 0, truncate(text, width), self.colors["header"] | curses.A_BOLD)
        self.addstr(
            0,
            min(display_width(text), width - 1),
            " " * (width - min(display_width(text), width)),
            self.colors["header"],
        )

    def draw_footer(self, text: str) -> None:
        height, width = self.stdscr.getmaxyx()
        self.addstr(height - 1, 0, truncate(text, width), curses.A_REVERSE)

    # -- list view ---------------------------------------------------------

    def draw_list(self) -> None:
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        records = self.records[self.file_idx]
        path = self.files[self.file_idx]
        self.draw_header(f" {path}  [{self.file_idx + 1}/{len(self.files)}]  {len(records)} records")
        self.draw_footer(LIST_FOOTER)

        match_set = set(self.matches)
        body_rows = max(1, height - 2)
        for row in range(body_rows):
            idx = self.top + row
            if idx >= len(records):
                break
            text, attr = summarize(records[idx], idx, width)
            if idx in match_set:
                text = "* " + text
            if idx == self.cursor:
                attr |= curses.A_REVERSE
            self.addstr(row + 1, 0, truncate(text, width), attr)

        if not records:
            self.addstr(1, 0, "(no records)", 0)

    def scroll_to_cursor(self) -> None:
        height, _ = self.stdscr.getmaxyx()
        body_rows = max(1, height - 2)
        if self.cursor < self.top:
            self.top = self.cursor
        elif self.cursor >= self.top + body_rows:
            self.top = self.cursor - body_rows + 1

    def set_cursor(self, idx: int) -> None:
        self.cursor = max(0, min(idx, len(self.records[self.file_idx]) - 1))
        self.scroll_to_cursor()

    # -- search ------------------------------------------------------------

    def search_text(self, idx: int) -> str:
        cache = self._search_cache[self.file_idx]
        if cache[idx] is None:
            cache[idx] = record_search_text(self.records[self.file_idx][idx]).lower()
        return cache[idx]

    def find_matches(self, query: str) -> list[int]:
        q = query.lower()
        return [i for i in range(len(self.records[self.file_idx])) if q in self.search_text(i)]

    def jump_match(self, step: int) -> None:
        if not self.query:
            return
        if not self.matches:
            self.matches = self.find_matches(self.query)
        if not self.matches:
            return
        self.match_idx = (self.match_idx + step) % len(self.matches)
        self.set_cursor(self.matches[self.match_idx])

    def prompt(self, title: str) -> str | None:
        """Read a line of input in the footer; returns None on cancel."""
        height, width = self.stdscr.getmaxyx()
        buf = ""
        curses.curs_set(1)
        try:
            while True:
                self.addstr(height - 1, 0, " " * (width - 1))
                self.addstr(height - 1, 0, truncate(f"{title}{buf}", width), curses.A_REVERSE)
                self.stdscr.move(height - 1, min(display_width(f"{title}{buf}"), width - 1))
                self.stdscr.refresh()
                ch = self.stdscr.getch()
                if ch in (curses.KEY_ENTER, 10, 13):
                    return buf
                if ch == ESC:
                    return None
                if ch in (curses.KEY_BACKSPACE, 127, 8):
                    buf = buf[:-1]
                elif 32 <= ch < 127:
                    buf += chr(ch)
        finally:
            curses.curs_set(0)

    # -- detail view -------------------------------------------------------

    def draw_detail(self, lines: list[tuple[str, int]], offset: int) -> None:
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        records = self.records[self.file_idx]
        self.draw_header(f" {self.files[self.file_idx]}  record {self.cursor}/{len(records) - 1}")
        self.draw_footer(DETAIL_FOOTER)

        body_rows = max(1, height - 2)
        for row in range(body_rows):
            idx = offset + row
            if idx >= len(lines):
                break
            text, attr = lines[idx]
            self.addstr(row + 1, 0, truncate(text, width), attr)

    def view_record(self) -> None:
        records = self.records[self.file_idx]
        if not records:
            return
        offset = 0
        while True:
            _, width = self.stdscr.getmaxyx()
            lines = detail_lines(records[self.cursor], max(1, width))
            self.draw_detail(lines, offset)
            ch = self.stdscr.getch()
            height, _ = self.stdscr.getmaxyx()
            body_rows = max(1, height - 2)
            if ch in (ord("q"), ESC, curses.KEY_ENTER, 10, 13):
                return
            elif ch in (ord("k"), curses.KEY_UP):
                offset = max(0, offset - 1)
            elif ch in (ord("j"), curses.KEY_DOWN):
                offset = min(max(0, len(lines) - body_rows), offset + 1)
            elif ch == curses.KEY_PPAGE:
                offset = max(0, offset - body_rows)
            elif ch == curses.KEY_NPAGE:
                offset = min(max(0, len(lines) - body_rows), offset + body_rows)
            elif ch in (ord("g"), curses.KEY_HOME):
                offset = 0
            elif ch in (ord("G"), curses.KEY_END):
                offset = max(0, len(lines) - body_rows)
            elif ch == curses.KEY_RESIZE:
                pass  # redraw with new size next iteration

    # -- help overlay ------------------------------------------------------

    def show_help(self) -> None:
        height, width = self.stdscr.getmaxyx()
        box_w = max(len(line) for line in HELP_TEXT) + 4
        box_h = len(HELP_TEXT) + 2
        y = max(0, (height - box_h) // 2)
        x = max(0, (width - box_w) // 2)
        for i in range(box_h):
            self.addstr(y + i, x, " " * box_w, curses.A_REVERSE)
        for i, line in enumerate(HELP_TEXT):
            self.addstr(y + 1 + i, x + 2, line, curses.A_REVERSE)
        self.stdscr.getch()

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        while True:
            self.draw_list()
            ch = self.stdscr.getch()
            records = self.records[self.file_idx]
            height, _ = self.stdscr.getmaxyx()
            page = max(1, height - 2)

            if ch in (ord("q"), ord("Q")):
                return
            elif ch in (ord("k"), curses.KEY_UP):
                self.set_cursor(self.cursor - 1)
            elif ch in (ord("j"), curses.KEY_DOWN):
                self.set_cursor(self.cursor + 1)
            elif ch == curses.KEY_PPAGE:
                self.set_cursor(self.cursor - page)
            elif ch == curses.KEY_NPAGE:
                self.set_cursor(self.cursor + page)
            elif ch in (ord("g"), curses.KEY_HOME):
                self.set_cursor(0)
            elif ch in (ord("G"), curses.KEY_END):
                self.set_cursor(len(records) - 1)
            elif ch in (curses.KEY_ENTER, 10, 13):
                self.view_record()
            elif ch == ord("/"):
                query = self.prompt("search: ")
                if query:
                    self.query = query
                    self.matches = self.find_matches(query)
                    self.match_idx = 0
                    if self.matches:
                        self.set_cursor(self.matches[0])
            elif ch == ord("n"):
                self.jump_match(1)
            elif ch == ord("N"):
                self.jump_match(-1)
            elif ch == ord("\t") or ch == curses.KEY_BTAB:
                if len(self.files) > 1:
                    self.file_idx = (self.file_idx + 1) % len(self.files)
                    self.cursor = 0
                    self.top = 0
                    self.matches = []
                    self.match_idx = -1
            elif ch == ord("h"):
                self.show_help()
            elif ch == curses.KEY_RESIZE:
                pass  # redraw with new size next iteration


def print_plain(files: list[str]) -> None:
    """Non-interactive fallback: pretty-print every record to stdout."""
    for path in files:
        print(f"===== {path} =====")
        for index, record in enumerate(load_jsonl(path)):
            metric = record.get("metric", "-")
            finish_reason = record.get("finish_reason") or "-"
            tokens = " / ".join(str(record.get(k, "-")) for k in ("input_tokens", "output_tokens", "total_tokens"))
            print(
                f"[{index}] id: {record.get('id', '?')}  metric: {metric}  finish_reason: {finish_reason}  tokens: {tokens}"
            )
            for msg in record.get("messages") or []:
                print(f"  ### {msg.get('role', '?')}")
                print(msg.get("content", ""))
            print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="JSONL file(s) to view")
    parser.add_argument("--plain", action="store_true", help="print records to stdout instead of launching the TUI")
    args = parser.parse_args()

    if args.plain:
        print_plain(args.files)
        return 0

    try:
        curses.wrapper(lambda stdscr: App(stdscr, args.files).run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
