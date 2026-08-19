"""Static server for the passenger app.

Same no-cache handling as the Twin's serve.py, and for the same reason: the
stdlib server sends Last-Modified with no Cache-Control, so a browser keeps
serving the copy it already has and edits look like they silently reverted.
HTML and the generated JSON are always fetched fresh; nothing else here is
big enough for caching to matter.

Usage (from this directory):
    python serve.py [port]        # default 8090
"""

import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent


class NoCache(SimpleHTTPRequestHandler):
    def _fresh(self) -> bool:
        path = self.path.split("?", 1)[0]
        return path.endswith((".html", ".json", "/")) or path == ""

    def send_head(self):
        if self._fresh() and "If-Modified-Since" in self.headers:
            del self.headers["If-Modified-Since"]
        return super().send_head()

    def end_headers(self):
        if self._fresh():
            self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    handler = partial(NoCache, directory=str(HERE))
    print(f"[user-app] serving {HERE} on http://127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    main()
