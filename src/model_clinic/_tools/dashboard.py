"""Dashboard: Serve an interactive model health report in a local browser.

Usage:
    model-clinic dashboard checkpoint.pt
    model-clinic dashboard checkpoint.pt --port 8080
    model-clinic dashboard model_name --hf
    model-clinic dashboard checkpoint.pt --no-browser
"""

import argparse
import os
import re
import sys
import tempfile
import threading
import webbrowser
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler


def _auto_filename(model_path):
    """Generate report filename from model name."""
    base = os.path.basename(model_path)
    name = re.sub(r'\.(pt|pth|safetensors|bin|ckpt)$', '', base)
    name = re.sub(r'[^\w\-.]', '_', name)
    if len(name) > 60:
        name = name[:60]
    return f"{name}_dashboard.html"


class _SingleFileHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves a single HTML file for all GET requests."""

    report_path = None  # Set before server starts
    report_bytes = None

    def do_GET(self):
        if self.report_bytes is None:
            self.send_error(500, "Report not loaded")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(self.report_bytes)))
        self.end_headers()
        self.wfile.write(self.report_bytes)

    def log_message(self, format, *args):
        # Suppress per-request logs; dashboard prints its own status line
        pass


def main():
    parser = argparse.ArgumentParser(
        prog="model-clinic dashboard",
        description="Serve an interactive model health dashboard on localhost",
    )
    parser.add_argument("model", help="Path to .pt checkpoint or HF model name")
    parser.add_argument("--port", "-p", type=int, default=7860,
                        help="Port to serve on (default: 7860)")
    parser.add_argument("--hf", action="store_true",
                        help="Load as HuggingFace model")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not auto-open browser")
    parser.add_argument("--debug", action="store_true",
                        help="Include raw tensor stats in the report")
    args = parser.parse_args()

    from model_clinic._loader import load_state_dict, build_meta
    from model_clinic.clinic import diagnose, prescribe
    from model_clinic._health_score import compute_health_score
    from model_clinic._report import generate_report

    print(f"Loading: {args.model}")
    state_dict, raw_meta = load_state_dict(args.model, hf=args.hf)
    meta = build_meta(state_dict, source=raw_meta.get("source", "unknown"),
                      extra=raw_meta)

    print("Diagnosing...")
    findings = diagnose(state_dict, meta=raw_meta)
    prescriptions = prescribe(findings)
    health = compute_health_score(findings)

    # Write to a temp file then read back as bytes
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    tmp.close()
    report_path = tmp.name

    print("Building interactive report...")
    generate_report(
        state_dict, findings, prescriptions, health, meta,
        report_path,
        interactive=True,
        debug=args.debug,
    )

    with open(report_path, "rb") as f:
        report_bytes = f.read()

    # Clean up temp file (we hold the content in memory)
    try:
        os.unlink(report_path)
    except OSError:
        pass

    # Inject a small banner at top of page noting it is the live dashboard
    banner = (
        b'<div style="position:fixed;top:0;left:0;right:0;z-index:9999;'
        b'background:#0d2040;border-bottom:1px solid #2a4a7a;padding:.4rem 1rem;'
        b'font-size:.78rem;color:#64b5f6;font-family:sans-serif;display:flex;'
        b'justify-content:space-between;align-items:center">'
        b'<span>model-clinic dashboard &mdash; live</span>'
        b'<span style="color:#555">Ctrl+C in terminal to stop</span></div>'
        b'<div style="height:2rem"></div>'
    )
    report_bytes = report_bytes.replace(b"<body>", b"<body>" + banner, 1)

    _SingleFileHandler.report_bytes = report_bytes

    # Find a free port if the requested one is busy
    port = args.port
    for attempt in range(5):
        try:
            server = HTTPServer(("127.0.0.1", port), _SingleFileHandler)
            break
        except OSError:
            port += 1
    else:
        print(f"Could not bind to port {args.port} or nearby ports.", file=sys.stderr)
        sys.exit(1)

    url = f"http://localhost:{port}"
    print(f"\nDashboard running at {url}  (Ctrl+C to stop)\n")
    print(f"  Health: {health.overall}/100  Grade: {health.grade}")
    print(f"  Findings: {len(findings)}  |  {meta.num_params:,} params\n")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
