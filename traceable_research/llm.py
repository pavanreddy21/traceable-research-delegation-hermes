"""LLM client for any OpenAI-compatible endpoint, local embeddings, and a record/replay cache.

Every LLM, embedding and HTTP call is keyed by a hash of its request and stored under cache/.
That makes runs reproducible: `TRACE_MODE=replay` re-runs the saved evaluation with no key and no network.
"""
import base64
import hashlib
import json
import re
import threading
import time

import numpy as np
import requests

from . import config


class ReplayMiss(RuntimeError):
    pass


class Usage:
    """Thread-safe counters for one task (cost and latency reporting)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.calls = 0
        self.cached = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0

    def add(self, usage, seconds, cached):
        with self.lock:
            self.calls += 1
            self.cached += int(cached)
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.seconds += seconds

    def as_dict(self):
        return {"llm_calls": self.calls, "cached_calls": self.cached, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "llm_seconds": round(self.seconds, 1)}


USAGE = Usage()


def _key(kind, payload):
    blob = json.dumps({"kind": kind, **payload}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def cached(kind, payload, fetch):
    """Return (value, was_cached). fetch() is called only on a cache miss in auto/live mode."""
    path = config.CACHE_DIR / kind / f"{_key(kind, payload)}.json"
    if config.MODE != "live" and path.exists():
        return json.loads(path.read_text()), True
    if config.MODE == "replay":
        raise ReplayMiss(f"no recorded {kind} response for this request (run once with a key, mode=auto)")
    value = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1))
    return value, False


_RATE_LOCK = threading.Lock()
_NEXT_SLOT = [0.0]


def _throttle():
    """Global requests-per-minute limit shared by all agent threads (LLM_RPM, default 5)."""
    gap = 60.0 / max(1, int(config.os.environ.get("LLM_RPM", "5")))
    with _RATE_LOCK:
        wait = _NEXT_SLOT[0] - time.time()
        _NEXT_SLOT[0] = max(time.time(), _NEXT_SLOT[0]) + gap
    if wait > 0:
        time.sleep(wait)


def _post(path, body, timeout=180, attempts=12):
    if not (config.LLM_BASE_URL and config.LLM_API_KEY):
        raise RuntimeError("Set LLM_BASE_URL and LLM_API_KEY in .env, or use TRACE_MODE=replay")
    for attempt in range(attempts):
        _throttle()
        sent = time.time()
        try:
            resp = requests.post(f"{config.LLM_BASE_URL}/{path}", json=body, timeout=timeout,
                                 headers={"Authorization": f"Bearer {config.LLM_API_KEY}"})
        except requests.RequestException as err:
            if attempt == attempts - 1:
                raise RuntimeError(f"LLM endpoint unreachable: {err}") from err
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
            retry_after = resp.headers.get("Retry-After", "")
            time.sleep(float(retry_after) if retry_after.replace(".", "").isdigit() else min(60, 5 * 2 ** attempt))
            continue
        if resp.status_code >= 400:
            raise RuntimeError(f"LLM endpoint {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        data["_seconds"] = round(time.time() - sent, 2)  # request latency only, not rate-limit waiting
        return data


def _post_window(body):
    """Some models on the gateway only accept one completion window; retry with the one the error names."""
    try:
        return _post("chat/completions", body)
    except RuntimeError as err:
        m = re.search(r"this model supports: (\w+)", str(err))
        if not m:
            raise
        return _post("chat/completions", dict(body, metadata={"completion_window": m.group(1)}))


def chat(messages, model=None, max_tokens=900, temperature=0.0):
    """max_tokens is the visible-answer budget; reasoning models get extra room for their hidden reasoning."""
    model = model or config.AGENT_MODEL
    body = {"model": model, "messages": messages, "max_tokens": max_tokens + 3000, "temperature": temperature,
            "metadata": {"completion_window": "balanced"}}
    data, was_cached = cached("llm", body, lambda: _post_window(body))
    USAGE.add(data.get("usage", {}), data.get("_seconds", 0.0), was_cached)
    msg = data["choices"][0]["message"]
    text = msg.get("content") or ""
    if isinstance(text, list):  # some models return content parts
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


_EMBEDDERS = {}
_EMBED_LOCK = threading.Lock()


def _local_embed(model, texts):
    from fastembed import TextEmbedding  # local ONNX model, downloaded once (~70-220 MB)
    with _EMBED_LOCK:
        if model not in _EMBEDDERS:
            _EMBEDDERS[model] = TextEmbedding(model_name=model)
        vecs = np.array(list(_EMBEDDERS[model].embed(texts)), dtype=np.float16)
    return {"shape": list(vecs.shape), "b64": base64.b64encode(vecs.tobytes()).decode()}


def embed(texts, model=None):
    """Sentence embeddings, cached as float16 so replays never need the model."""
    model = model or config.EMBED_MODEL
    out = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        data, _ = cached("embed", {"model": model, "input": batch}, lambda: _local_embed(model, batch))
        arr = np.frombuffer(base64.b64decode(data["b64"]), dtype=np.float16).reshape(data["shape"])
        out.extend(arr.astype(np.float32))
    return out


def parse_json(text):
    """Pull the first JSON object or array out of a model reply. Raises ValueError if none parses."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON object in reply")
    start = min(starts)
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text[start:])
        return value
    except json.JSONDecodeError as err:
        raise ValueError(f"invalid JSON: {err}") from err
