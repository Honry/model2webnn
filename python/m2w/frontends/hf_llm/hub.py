"""Get a Hugging Face model onto the disk: a local directory is used as is, a repo id is downloaded (config, tokenizer files, safetensors shards).

The endpoint is configurable (``--endpoint`` / ``HF_ENDPOINT``, e.g. https://hf-mirror.com) because huggingface.co is not reachable from every network.
Plain HTTPS with Range-resume, no dependency on huggingface_hub (it is used for ``publish`` only, later)."""
from __future__ import annotations

import json
import http.client
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

SMALL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
               "chat_template.json", "vocab.json", "merges.txt", "preprocessor_config.json")
DEFAULT_ENDPOINT = "https://huggingface.co"


def _open(url: str, token: Optional[str], start: int = 0):
    request = urllib.request.Request(url, headers={"User-Agent": "m2w"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if start:
        request.add_header("Range", f"bytes={start}-")
    for attempt in range(8):   # mirrors rate-limit (429) and hiccup (5xx) now and then: back off and retry
        try:
            return urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 7:
                raise
            delay = float(e.headers.get("Retry-After") or min(60, 2 ** (attempt + 1)))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 7:
                raise
            delay = min(60, 2 ** (attempt + 1))
        time.sleep(delay)


def _download(url: str, dest: Path, token: Optional[str], progress: Callable[[str], None], expected: Optional[int] = None) -> None:
    """Download to dest.part and rename when complete; any network error resumes from the bytes already on disk (Range)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected is None or dest.stat().st_size == expected):   # dest only appears once a download completed
        return
    part = dest.with_name(dest.name + ".part")
    for attempt in range(12):
        start = part.stat().st_size if part.exists() else 0
        try:
            try:
                response = _open(url, token, start)
            except urllib.error.HTTPError as e:
                if e.code == 416 and start:     # the .part file is already complete
                    part.replace(dest)
                    return
                raise
            if response.status != 206:
                start = 0
            total = int(response.headers.get("Content-Length", 0)) + start
            done, last = start, start
            with open(part, "ab" if start else "wb") as f:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total and done - last >= 64 << 20:
                        last = done
                        progress(f"{dest.name}: {done / 2**20:.0f} / {total / 2**20:.0f} MB")
            part.replace(dest)
            return
        except (TimeoutError, ConnectionError, urllib.error.URLError, http.client.IncompleteRead) as e:
            if attempt == 11:
                raise
            progress(f"{dest.name}: {type(e).__name__}, resuming at {part.stat().st_size / 2**20 if part.exists() else 0:.0f} MB")
            time.sleep(min(30, 2 ** (attempt + 1)))


def _download_parallel(url: str, dest: Path, token: Optional[str], progress: Callable[[str], None], total: int, connections: int = 8, segment: int = 32 << 20) -> None:
    """Big files: `connections` ranged requests at once (mirrors often throttle each connection). The file is preallocated and written in 32 MB segments;
    finished segments are recorded in dest.progress, so an interrupted run only repeats the segments that were in flight."""
    from concurrent.futures import ThreadPoolExecutor

    part, record = dest.with_name(dest.name + ".part"), dest.with_name(dest.name + ".progress")
    count = -(-total // segment)
    done = set(json.loads(record.read_text())["done"]) if record.exists() and part.exists() and part.stat().st_size == total else set()
    if not (part.exists() and part.stat().st_size == total):
        with open(part, "wb") as f:
            f.truncate(total)
        done = set()
    lock = __import__("threading").Lock()
    state = {"bytes": sum(min(segment, total - i * segment) for i in done), "last": 0}

    def fetch(index: int) -> None:
        start, end = index * segment, min(total, (index + 1) * segment) - 1
        for attempt in range(12):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "m2w", "Range": f"bytes={start}-{end}"})
                if token:
                    request.add_header("Authorization", f"Bearer {token}")
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise ConnectionError(f"server ignored the Range header (HTTP {response.status})")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise http.client.IncompleteRead(data)
                with lock:
                    with open(part, "r+b") as f:
                        f.seek(start)
                        f.write(data)
                    done.add(index)
                    record.write_text(json.dumps({"done": sorted(done)}), newline="\n")
                    state["bytes"] += len(data)
                    if state["bytes"] - state["last"] >= 64 << 20:
                        state["last"] = state["bytes"]
                        progress(f"{dest.name}: {state['bytes'] / 2**20:.0f} / {total / 2**20:.0f} MB ({connections} connections)")
                return
            except (TimeoutError, ConnectionError, urllib.error.URLError, http.client.IncompleteRead, urllib.error.HTTPError) as e:
                if attempt == 11 or (isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503, 504)):
                    raise
                time.sleep(min(30, 2 ** (attempt + 1)))

    with ThreadPoolExecutor(max_workers=connections) as pool:
        for future in [pool.submit(fetch, i) for i in range(count) if i not in done]:
            future.result()
    part.replace(dest)
    record.unlink(missing_ok=True)


def _size_of(url: str, token: Optional[str]) -> int:
    """Total size from a one-byte ranged request (0 if the server does not say)."""
    request = urllib.request.Request(url, headers={"User-Agent": "m2w", "Range": "bytes=0-0"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.headers.get("Content-Range", "/0").rsplit("/", 1)[-1] or 0)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return 0


def resolve_model(model: str, cache_dir=None, endpoint: Optional[str] = None, token: Optional[str] = None, revision: str = "main",
                  progress: Callable[[str], None] = lambda _m: None, weights: bool = True) -> Path:
    """-> a local directory with config.json and the safetensors files of ``model`` (a local path or an HF repo id like 'microsoft/Aion-1.0-Instruct-2.3B').
    weights=False downloads only the small config / tokenizer files (for ``inspect``)."""
    local = Path(model)
    if local.is_dir():
        return local
    endpoint = (endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
    token = token or os.environ.get("HF_TOKEN")
    target = Path(cache_dir or os.environ.get("M2W_CACHE") or Path.home() / ".cache" / "m2w") / model.replace("/", "--")
    api = json.loads(_open(f"{endpoint}/api/models/{model}/revision/{revision}", token).read())
    files = {s["rfilename"]: s for s in api["siblings"]}
    wanted = [n for n in files if n in SMALL_FILES or (weights and (n.endswith(".safetensors") or n == "model.safetensors.index.json"))
              or (n.endswith(".py") and "/" not in n)]
    if weights and not any(n.endswith(".safetensors") for n in wanted):
        raise FileNotFoundError(f"{model} has no .safetensors files (other formats are not supported yet)")
    for name in sorted(wanted):
        size = files[name].get("size")
        progress(f"downloading {name}" + (f" ({size / 2**20:.0f} MB)" if size else ""))
        url, dest = f"{endpoint}/{model}/resolve/{revision}/{name}", target / name
        if not dest.exists() and name.endswith(".safetensors") and (total := size or _size_of(url, token)) > 256 << 20:
            _download_parallel(url, dest, token, progress, total, connections=int(os.environ.get("M2W_CONNECTIONS", 8)))
        else:
            _download(url, dest, token, progress, size)
    return target
