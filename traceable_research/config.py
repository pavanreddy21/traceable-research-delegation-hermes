"""Settings, read from the environment or a local, gitignored .env file."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path=ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# Any OpenAI-compatible chat endpoint works (Cloudflare AI Gateway, OpenAI, Workers AI, a local Ollama...).
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

# Model choices come from experiments/README.md (E1 picks the checker, E2 the embedding model).
AGENT_MODEL = os.environ.get("AGENT_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash-1017")
CHECKER_MODEL = os.environ.get("CHECKER_MODEL", AGENT_MODEL)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5")  # local, via fastembed

# "auto": use the replay cache when a call was recorded, otherwise call live and record it.
# "replay": never touch the network (reviewers without a key). "live": always call, refresh cache.
MODE = os.environ.get("TRACE_MODE", "auto")
CACHE_DIR = Path(os.environ.get("TRACE_CACHE", ROOT / "cache"))

# Hard budgets for one research task (safety + cost bounds).
MAX_AGENT_STEPS = 6        # ReAct steps per role agent
MAX_ROUNDS = 3             # coordinator review rounds (Tree of Thought follow-ups)
MAX_BRANCHES = 3           # candidate actions proposed per decision
KEEP_BRANCHES = 2          # beam width
MAX_LLM_CALLS = 80         # whole task
MAX_SOURCES_PER_SEARCH = 5
