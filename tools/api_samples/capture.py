"""Fetches live responses from the free APIs CryptoPulse uses (or evaluates) and saves them as
samples, so test fixtures can be built from real payloads instead of documentation.

Run by .github/workflows/api-samples.yml (GitHub's network can reach every API; the dev sandbox
cannot). Usage: python capture.py urls.txt out_dir

urls.txt: one "name<TAB>url[<TAB>delay_seconds]" per line; '#' lines are comments. A url may
contain {name.jsonpath} placeholders filled from an earlier response, e.g.
{gt_trending.data.0.attributes.address}.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\.([^}]+)\}")


def lookup(doc, path: str):
    cur = doc
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def main() -> int:
    urls_file, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    docs: dict[str, object] = {}
    for raw in urls_file.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        cols = raw.split("\t")
        name, url = cols[0].strip(), cols[1].strip()
        delay = float(cols[2]) if len(cols) > 2 else 1.0
        try:
            url = PLACEHOLDER.sub(lambda m: str(lookup(docs[m.group(1)], m.group(2))), url)
        except Exception as exc:  # placeholder source missing or shaped differently
            print(f"=== {name}: SKIPPED (placeholder: {type(exc).__name__} {exc})")
            continue
        time.sleep(delay)
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json, text/html;q=0.9, */*;q=0.8"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status, body, ctype = resp.status, resp.read(), resp.headers.get("Content-Type", "")
                headers = {k: v for k, v in resp.headers.items() if k.lower().startswith(("x-ratelimit", "retry-after", "ratelimit", "cf-"))}
        except urllib.error.HTTPError as exc:
            status, body, ctype = exc.code, exc.read() or b"", exc.headers.get("Content-Type", "") if exc.headers else ""
            headers = dict(exc.headers.items()) if exc.headers else {}
        except Exception as exc:
            print(f"=== {name}: ERROR {type(exc).__name__}: {exc}\n    {url}")
            continue
        ms = int((time.monotonic() - started) * 1000)
        ext = "json" if "json" in ctype else "html" if "html" in ctype else "txt"
        (out_dir / f"{name}.{ext}").write_bytes(body)
        print(f"=== {name}: HTTP {status} {len(body)} bytes {ctype} {ms} ms\n    {url}\n    headers: {headers}")
        text = body.decode("utf-8", "replace")
        if ext == "json":
            try:
                doc = json.loads(text)
                docs[name] = doc
                print(preview(doc))
                continue
            except ValueError:
                pass
        print("    " + text[:1500].replace("\n", "\n    "))
    return 0


def preview(doc, limit: int = 3500) -> str:
    """First elements of every list, so the shape is visible without the whole payload."""

    def trim(x, depth=0):
        if isinstance(x, list):
            return [trim(v, depth + 1) for v in x[:2]] + ([f"... {len(x) - 2} more"] if len(x) > 2 else [])
        if isinstance(x, dict):
            return {k: trim(v, depth + 1) for k, v in list(x.items())[:60]}
        if isinstance(x, str) and len(x) > 160:
            return x[:160] + "..."
        return x

    out = json.dumps(trim(doc), indent=1, ensure_ascii=False)
    return out[:limit] + (f"\n... ({len(out)} chars)" if len(out) > limit else "")


if __name__ == "__main__":
    sys.exit(main())
