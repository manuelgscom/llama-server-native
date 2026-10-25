#!/usr/bin/env python3
"""Web search MCP server for llama-server.

stdio transport: one JSON-RPC message per line on stdin, the response on
stdout. Nothing else may go to stdout or the protocol breaks; diagnostic
messages go to stderr.

Standard library only, to keep pip and dependencies out of the image.

Tools:
  web_search(query, max_results)  -> queries the local SearXNG instance
  fetch_url(url, max_chars)       -> downloads a page and converts it to readable text
"""

import html
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")
TIMEOUT = float(os.environ.get("WEBSEARCH_TIMEOUT", "20"))
MAX_BYTES = 4 * 1024 * 1024          # download cap for fetch_url
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) llama-server-websearch/1.0"

PROTOCOL_VERSION = "2024-11-05"      # the one llama.cpp sends (server-mcp.cpp)


def log(msg):
    print(f"[mcp-websearch] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Readable text extraction
# ---------------------------------------------------------------------------

class TextExtractor(HTMLParser):
    """Extracts visible text, dropping script/style/nav/header/footer.

    Not real readability, but it removes most of the noise: menus,
    scripts and stylesheets. Good enough for the model to read an article.
    """

    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "header",
            "footer", "aside", "form", "button", "iframe"}
    BREAK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "section", "article", "blockquote", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip_depth = 0
        self.title = None
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title and self.title is None:
            self.title = data.strip()
        if self.skip_depth == 0:
            self.parts.append(data)

    def text(self):
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r" *\n *", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def http_get(url, accept="*/*"):
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": accept,
        "Accept-Language": "es,en;q=0.8",
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        raw = resp.read(MAX_BYTES)
        ctype = resp.headers.get("Content-Type", "")
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw, ctype, charset, resp.geturl()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def tool_web_search(args):
    query = (args.get("query") or "").strip()
    if not query:
        return "Error: missing 'query' parameter.", True

    n = args.get("max_results", 5)
    try:
        n = max(1, min(int(n), 15))
    except (TypeError, ValueError):
        n = 5

    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "language": args.get("language", "es"),
        "safesearch": "0",
    })
    url = f"{SEARXNG_URL}/search?{params}"

    try:
        raw, _, charset, _ = http_get(url, accept="application/json")
        data = json.loads(raw.decode(charset, errors="replace"))
    except urllib.error.URLError as e:
        return (f"Error querying SearXNG at {SEARXNG_URL}: {e}. "
                "Check that the 'searxng' service is running."), True
    except json.JSONDecodeError:
        return ("SearXNG did not return JSON. The 'json' format must be enabled "
                "in searxng/settings.yml."), True

    results = data.get("results") or []
    if not results:
        return f"No results for: {query}", False

    out = [f"Search results for: {query}\n"]
    for i, r in enumerate(results[:n], 1):
        title = (r.get("title") or "untitled").strip()
        link = (r.get("url") or "").strip()
        body = (r.get("content") or "").strip()
        out.append(f"{i}. {title}\n   {link}")
        if body:
            out.append(f"   {body}")
        out.append("")

    answers = data.get("answers") or []
    if answers:
        out.append("Direct answer: " + "; ".join(str(a) for a in answers))

    out.append("Use fetch_url with one of these URLs to read the full page.")
    return "\n".join(out), False


def tool_fetch_url(args):
    url = (args.get("url") or "").strip()
    if not url:
        return "Error: missing 'url' parameter.", True
    if not url.startswith(("http://", "https://")):
        return "Error: the URL must start with http:// or https://", True

    limit = args.get("max_chars", 20000)
    try:
        limit = max(500, min(int(limit), 100000))
    except (TypeError, ValueError):
        limit = 20000

    try:
        raw, ctype, charset, final_url = http_get(url, accept="text/html,*/*")
    except urllib.error.HTTPError as e:
        return f"HTTP error {e.code} while downloading {url}", True
    except urllib.error.URLError as e:
        return f"Could not download {url}: {e.reason}", True
    except Exception as e:  # timeouts, DNS, TLS...
        return f"Could not download {url}: {e}", True

    body = raw.decode(charset, errors="replace")

    if "html" in ctype.lower():
        parser = TextExtractor()
        try:
            parser.feed(body)
        except Exception:
            pass
        title = parser.title or ""
        text = parser.text()
    else:
        title = ""
        text = body.strip()

    if not text:
        return f"The page {final_url} has no extractable text.", False

    header = f"URL: {final_url}\n"
    if title:
        header += f"Title: {html.unescape(title)}\n"
    header += "\n"

    if len(text) > limit:
        text = text[:limit] + f"\n\n[...truncated at {limit} characters]"

    return header + text, False


TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Searches the internet and returns the title, URL and snippet of "
            "each result. Use it when you need current information, anything "
            "after your training cutoff, or facts you should verify."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "max_results": {"type": "integer",
                                "description": "Number of results (1-15, default 5)"},
                "language": {"type": "string",
                             "description": "Results language, e.g. 'es' or 'en'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Downloads a web page and returns its readable text. Use it "
            "after web_search to read a relevant result in full, or when "
            "the user gives you a specific URL."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL (http:// or https://)"},
                "max_chars": {"type": "integer",
                              "description": "Maximum number of characters to return (default 20000)"},
            },
            "required": ["url"],
        },
    },
]

HANDLERS = {"web_search": tool_web_search, "fetch_url": tool_fetch_url}


# ---------------------------------------------------------------------------
# JSON-RPC loop
# ---------------------------------------------------------------------------

def reply(msg_id, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def handle(req):
    method = req.get("method")
    msg_id = req.get("id")

    # Notifications have no id and get no response
    if msg_id is None:
        return

    if method == "initialize":
        reply(msg_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "websearch", "version": "1.0"},
        })

    elif method == "tools/list":
        reply(msg_id, {"tools": TOOLS})

    elif method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        fn = HANDLERS.get(name)
        if fn is None:
            reply(msg_id, error={"code": -32601, "message": f"unknown tool: {name}"})
            return
        try:
            text, is_error = fn(args)
        except Exception as e:
            text, is_error = f"Internal error in {name}: {e}", True
        reply(msg_id, {"content": [{"type": "text", "text": text}], "isError": is_error})

    elif method == "ping":
        reply(msg_id, {})

    else:
        reply(msg_id, error={"code": -32601, "message": f"unsupported method: {method}"})


def main():
    log(f"started, SearXNG at {SEARXNG_URL}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            handle(req)
        except Exception as e:
            log(f"error processing request: {e}")


if __name__ == "__main__":
    main()
