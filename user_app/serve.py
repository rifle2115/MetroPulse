"""Server for the passenger app: static files plus the crowd forecast API.

Static files get the same no-cache handling as the Twin's serve.py: the stdlib
server sends Last-Modified with no Cache-Control, so a browser keeps serving
the copy it already has and edits look like they silently reverted.

    GET /api/forecast?station=Central%20City&at=2026-11-08T14:00&hours=18

returns the forecaster's prediction (pipeline/prediction/forecaster.py). The
forecast exists only here — it is not published to MQTT, so the Control Room
and the Twin never receive it.

Listens on all network interfaces so a phone on the same Wi-Fi can open it.

Usage (from this directory, with the project venv so pandas/scikit-learn load):
    ..\\.venv\\Scripts\\python serve.py [port]        # default 8090
"""

import json
import socket
import sys
import threading
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline" / "prediction"))

_forecaster = None
_forecaster_error = None
_lock = threading.Lock()


def get_forecaster():
    global _forecaster, _forecaster_error
    with _lock:
        if _forecaster is None and _forecaster_error is None:
            try:
                from forecaster import Forecaster
                _forecaster = Forecaster()
            except Exception as e:  # missing model or packages: app still runs
                _forecaster_error = f"{type(e).__name__}: {e}"
                print(f"[user-app] forecaster unavailable — {_forecaster_error}", flush=True)
    return _forecaster


class Handler(SimpleHTTPRequestHandler):
    def _fresh(self) -> bool:
        path = self.path.split("?", 1)[0]
        return path.endswith((".html", ".json", ".webmanifest", "/", ".js")) or path == ""

    def send_head(self):
        if self._fresh() and "If-Modified-Since" in self.headers:
            del self.headers["If-Modified-Since"]
        return super().send_head()

    def end_headers(self):
        if self._fresh():
            self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def _json(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path not in ("/api/forecast", "/api/stations", "/api/plan"):
            return super().do_GET()

        f = get_forecaster()
        if f is None:
            return self._json(503, {"error": "forecast unavailable", "detail": _forecaster_error})
        q = parse_qs(url.query)

        if url.path == "/api/stations":
            return self._json(200, {"stations": f.station_list()})

        if url.path == "/api/plan":
            # Planner tab: any real station, any date and hour
            try:
                at = datetime.fromisoformat(q.get("at", [""])[0][:16])
            except ValueError:
                return self._json(400, {"error": "at must be YYYY-MM-DDTHH:MM"})
            try:
                return self._json(200, f.plan(q.get("station", [""])[0], at))
            except KeyError as e:
                return self._json(404, {"error": str(e)})
            except Exception as e:
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

        try:
            at = datetime.fromisoformat(q.get("at", [""])[0][:16])
        except ValueError:
            at = datetime.now()
        hours = max(6, min(48, int(q.get("hours", ["18"])[0] or 18)))
        try:
            self._json(200, f.forecast(q.get("station", [None])[0], at, hours))
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):  # keep the console quiet except for API calls
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the LAN interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    handler = partial(Handler, directory=str(HERE))
    threading.Thread(target=get_forecaster, daemon=True).start()  # warm the model
    print(f"[user-app] on this PC:   http://127.0.0.1:{port}", flush=True)
    print(f"[user-app] on your phone: http://{lan_ip()}:{port}  (same Wi-Fi)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever()


if __name__ == "__main__":
    main()
