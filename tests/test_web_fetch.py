"""web_fetch + summarize_page tool tests (2026-08-22).

Regression: web_fetch returned only the first 8000 chars of any page with NO
marker and NO continuation, so summarizing long articles produced answers like
"the article was cut off before the full entry" (cache id 270).

Fix:
- web_fetch: whole-site single fetch (~100K chars) + explicit
  '[Content truncated — Call web_fetch(url, start=N) to continue.]' marker.
- summarize_page: fetches the ENTIRE page, compresses it server-side via
  parallel cheap-model chunk condensation, returns a short digest.

Run: .venv/bin/python tests/test_web_fetch.py
"""
import asyncio
import http.server
import re
import socketserver
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm

# The raw function is the actual decorated callable; the FunctionTool wrapper
# (schema/invoke plumbing) is exercised separately via params_json_schema.
fetch = llm.web_fetch.on_invoke_tool._get_wrapped_callable()

LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 3000
)  # ~135K chars — spans two 100K chunks
SHORT_TEXT = "<html><head><script>bad()</script><style>.x{}</style></head><body><h1>Title</h1><p>Hello world.</p></body></html>"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = LONG_TEXT.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class ShortHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = SHORT_TEXT.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def serve(handler_cls):
    srv = socketserver.TCPServer(("127.0.0.1", 0), handler_cls)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/page"


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def test_web_fetch():
    print("== web_fetch ==")
    srv, url = serve(Handler)
    try:
        c1 = fetch(url, 0)
        check("chunk1 length <= 100K + marker", len(c1) <= 100200, f"len={len(c1)}")
        check("chunk1 has truncation marker", "[Content truncated" in c1)
        m = re.search(r"start=(\d+)", c1)
        check("marker carries continuation offset", m and int(m.group(1)) == 100000,
              f"marker={m.group(0) if m else None}")
        c2 = fetch(url, 100000)
        check("chunk2 has no marker (page fits)", "[Content truncated" not in c2,
              repr(c2[-120:]))
        check("chunk1+chunk2 cover the page", len(c1) + len(c2) >= len(LONG_TEXT))
        ce = fetch(url, len(LONG_TEXT) + 1000)
        check("past-end returns end marker", "End of page content" in ce, repr(ce[:80]))
        mid = 12345
        check("mid chunk starts at offset", fetch(url, mid).startswith(LONG_TEXT[mid:mid + 40]),
              repr(fetch(url, mid)[:60]))
    finally:
        srv.shutdown()
        srv.server_close()

    srv2, url2 = serve(ShortHandler)
    try:
        c = fetch(url2, 0)
        check("scripts/styles/tags stripped", "bad()" not in c and ".x{" not in c,
              repr(c[:120]))
        check("short page no marker", "[Content truncated" not in c)
        check("short page text present", "Hello world." in c)
    finally:
        srv2.shutdown()
        srv2.server_close()

    sch = llm.web_fetch.params_json_schema
    props = sch.get("properties", {})
    check("schema start:integer default 0",
          props.get("start", {}).get("type") == "integer" and props.get("start", {}).get("default") == 0,
          str(props.get("start")))


async def _run_summarize(url, **kw):
    # summarize_page is a FunctionTool; reach the raw async fn
    raw = llm.summarize_page.on_invoke_tool._get_wrapped_callable()
    return raw(url, **kw)


def test_summarize_page():
    print("== summarize_page ==")
    # Long page → local extractive compression, digest ≤ max_chars
    srv, url = serve(Handler)
    try:
        out = asyncio.run(_run_summarize(url, max_chars=4000))
        check("digest ≤ max_chars", len(out) <= 4010, f"len={len(out)}")
        check("digest non-empty", len(out) > 200, f"len={len(out)}")
        check("no truncation marker", "[Content truncated" not in out)

        out_big = asyncio.run(_run_summarize(url, max_chars=20000))
        check("max_chars clamp to 20000", len(out_big) <= 20010, f"len={len(out_big)}")
        out_small = asyncio.run(_run_summarize(url, max_chars=1))
        # floor to 1000: a 1-char request must NOT collapse the digest
        check("max_chars floor prevents tiny trim", len(out_small) > 100,
              f"len={len(out_small)}")
    finally:
        srv.shutdown()
        srv.server_close()

    # Short page → returned as-is (no compression needed)
    srv2, url2 = serve(ShortHandler)
    try:
        out = asyncio.run(_run_summarize(url2, max_chars=4000))
        check("short page returned as-is", "Hello world." in out)
    finally:
        srv2.shutdown()
        srv2.server_close()

    # Compression failure → raw excerpt fallback, still no crash
    srv3, url3 = serve(Handler)
    orig = llm._sumy_digest
    llm._sumy_digest = lambda text, target: (_ for _ in ()).throw(RuntimeError("no"))
    try:
        out = asyncio.run(_run_summarize(url3, max_chars=4000))
    finally:
        llm._sumy_digest = orig
        srv3.shutdown()
        srv3.server_close()
    check("failure falls back to raw excerpt", "raw excerpt" in out and "fox jumps" in out,
          out[:150])

    sch = llm.summarize_page.params_json_schema
    props = sch.get("properties", {})
    check("summarize_page schema: url + max_chars",
          props.get("url", {}).get("type") == "string" and props.get("max_chars", {}).get("type") == "integer",
          str(props))

    check("summarize_page in ALL_TOOLS", any(getattr(t, "name", "") == "summarize_page" for t in llm.ALL_TOOLS))


def main():
    test_web_fetch()
    test_summarize_page()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
