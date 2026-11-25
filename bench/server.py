#!/usr/bin/env python
"""Static server for the NPU split bench (http://localhost:8123/bench/bench.html).

 - serves the repository root (bench/, lib/ = the `npm run build:lib` output that the pages import, ...) with COOP/COEP + CORS headers
 - POST /log   : append the request body (one JSON object) as a line to results/<run>.jsonl
 - GET  /last  : return the last line of the current results file (used by run_bench.ps1 to wait for 'done')
"""
import http.server
import os
import socketserver
import sys
import time

BENCH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BENCH)
RESULTS = os.path.join(BENCH, "results")
os.makedirs(RESULTS, exist_ok=True)
RUN = os.environ.get("BENCH_RUN") or time.strftime("run_%Y%m%d_%H%M%S")
LOGFILE = os.path.join(RESULTS, RUN + ".jsonl")


QWEN_DIR = os.environ.get("BENCH_QWEN_DIR")   # the Qwen3-4B static-capable ONNX directory, served as /qwen3/<file> (bench.html only)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def translate_path(self, path):
        # /qwen3/<file> -> the real Qwen3-4B static-capable ONNX directory
        if path.startswith("/qwen3/") and QWEN_DIR:
            return os.path.join(QWEN_DIR, os.path.basename(path.split("?")[0]))
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path.startswith("/log"):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n).decode("utf-8", "replace").replace("\n", " ")
            with open(LOGFILE, "a", encoding="utf-8") as f:
                f.write(body + "\n")
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path.startswith("/last"):
            last = ""
            if os.path.exists(LOGFILE):
                with open(LOGFILE, encoding="utf-8") as f:
                    lines = f.read().splitlines()
                    last = lines[-1] if lines else ""
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write((last or "{}").encode())
        else:
            super().do_GET()

    def log_message(self, fmt, *args):
        if "/log" not in str(args[0] if args else ""):
            sys.stderr.write("%s\n" % (fmt % args))


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    port = int(os.environ.get("BENCH_PORT", "8123"))
    print(f"serving {ROOT} on http://localhost:{port}  results -> {LOGFILE}", flush=True)
    Server(("127.0.0.1", port), Handler).serve_forever()
