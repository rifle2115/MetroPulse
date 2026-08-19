"""Static server for the Twin that doesn't let the browser cache index.html.

`python -m http.server` sends Last-Modified and no Cache-Control, so a browser
keeps serving the copy it already has and will not even revalidate. Editing
index.html then looks like nothing happened: the page still runs the old build,
which shows up as "the fix was reverted" rather than as a caching problem —
three separate changes appearing to vanish at once. `/` and `/index.html` are
cached as different entries too, so a cache-buster on one leaves the other
stale, which is how it survives a reload.

That is what the ?v=nocache1, ?v=fix3, ?v=perf1, ?v=final trail in server.log
was working around. This removes the need for it: the HTML is always fetched
fresh, while the character models — which are megabytes and never change — are
still cached normally.

Usage (from this directory):
    python serve.py [port]
"""

import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent


class NoCacheHTML(SimpleHTTPRequestHandler):
    def _is_html(self) -> bool:
        path = self.path.split("?", 1)[0]
        return path.endswith((".html", "/")) or path == ""

    def send_head(self):
        # SimpleHTTPRequestHandler answers 304 off If-Modified-Since. Dropping
        # the header for HTML means it can't, so a stale copy can never be
        # revalidated back into use.
        if self._is_html() and "If-Modified-Since" in self.headers:
            del self.headers["If-Modified-Since"]
        return super().send_head()

    def end_headers(self):
        if self._is_html():
            self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    handler = partial(NoCacheHTML, directory=str(HERE))
    print(f"[twin] serving {HERE} on http://127.0.0.1:{port} (index.html never cached)", flush=True)
    HTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    main()
