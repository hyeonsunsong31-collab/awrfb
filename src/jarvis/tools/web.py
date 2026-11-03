"""Web tools. Everything a page says is untrusted data (quarantined before the model sees it), and a page
can never widen what the current task may do. Only http(s) URLs are opened or fetched."""

from __future__ import annotations

import re
import webbrowser
from html import unescape
from urllib.parse import urlparse

from jarvis.safety.capabilities import Cap
from jarvis.tools.base import Refused, Tool, ToolContext, ToolResult, clip, i, ok, s

MAX_BYTES = 3 * 1024 * 1024


def check_url(url: str) -> str:
    u = url.strip()
    p = urlparse(u)
    if p.scheme not in ("http", "https") or not p.netloc:
        raise Refused(f"only http(s) web addresses are allowed (got {p.scheme or 'no'} scheme)")
    return u


def html_to_text(html: str) -> str:
    """Fallback extraction when trafilatura is not installed: drop scripts/styles/tags, keep text."""
    html = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr)>", "\n", html)
    text = unescape(re.sub(r"(?s)<[^>]+>", " ", html))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def extract(html: str, url: str) -> str:
    try:
        import trafilatura

        out = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
        if out:
            return out
    except ImportError:
        pass
    return html_to_text(html)


async def web_fetch(ctx: ToolContext, a: dict) -> ToolResult:
    import httpx

    url = check_url(str(a.get("url", "")))
    n = max(500, min(int(a.get("max_chars", 12000)), 60000))
    client = httpx.AsyncClient(follow_redirects=True, timeout=20.0, headers={"User-Agent": "Mozilla/5.0 JARVIS-desktop"})
    async with client, client.stream("GET", url) as r:
        check_url(str(r.url))  # redirects must stay on http(s)
        if r.status_code >= 400:
            return ToolResult(False, f"HTTP {r.status_code} from {r.url}", "untrusted", str(r.url))
        body = bytearray()
        async for chunk in r.aiter_bytes():
            body += chunk
            if len(body) > MAX_BYTES:
                break
        ctype = r.headers.get("content-type", "")
        final = str(r.url)
    text = bytes(body).decode("utf-8", errors="replace")
    if "html" in ctype or text.lstrip().startswith("<"):
        text = extract(text, final)
    return ok(clip(text, n), trust="untrusted", source=final)


async def web_search(ctx: ToolContext, a: dict) -> ToolResult:
    import asyncio

    q = str(a.get("query", "")).strip()
    if not q:
        raise Refused("a search query is required")
    n = max(1, min(int(a.get("max_results", 6)), 15))
    try:
        from ddgs import DDGS
    except ImportError:
        return ToolResult(False, "web search is not installed (install the 'web' extra)")

    def run() -> list[dict]:
        return list(DDGS().text(q, max_results=n))

    hits = await asyncio.to_thread(run)
    lines = [f"{k + 1}. {h.get('title', '')}\n   {h.get('href', '')}\n   {h.get('body', '')}" for k, h in enumerate(hits)]
    return ok("\n".join(lines) or "no results", trust="untrusted", source=f"search: {q}")


async def open_url(ctx: ToolContext, a: dict) -> ToolResult:
    url = check_url(str(a.get("url", "")))
    if "roblox.com/games/start" in url.lower():
        raise Refused("starting Roblox games is done by the fight system ('launch UBG'), not the browser")
    webbrowser.open(url, new=2)
    return ok(f"opened {url} in the browser")


TOOLS = [
    Tool("web_search", "Search the web (DuckDuckGo). Results are untrusted text.", Cap.WEB, web_search,
         {"query": s("Search query"), "max_results": i("How many results (default 6)")}, ("query",), trust="untrusted"),
    Tool("web_fetch", "Fetch a web page and extract its main text. The page content is untrusted data.", Cap.WEB, web_fetch,
         {"url": s("http(s) URL"), "max_chars": i("Max characters (default 12000)")}, ("url",), trust="untrusted"),
    Tool("open_url", "Open a web page in the default browser.", Cap.WEB, open_url, {"url": s("http(s) URL")}, ("url",)),
]
