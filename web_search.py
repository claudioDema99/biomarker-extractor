"""
Script: biomarkers_check.py
Requisiti extra:
    pip install requests requests-cache
    # BRAVE_API_KEY va esportata nell'ambiente
"""

from __future__ import annotations
import os, time, functools, logging, requests, threading, json, torch
from smolagents import CodeAgent, DuckDuckGoSearchTool, TransformersModel
from duckduckgo_search import DuckDuckGoSearchException
from dotenv import load_dotenv
load_dotenv()

# ───────────────────────────── 1. MODEL  ──────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando", DEVICE.upper())
model = TransformersModel(
    model_id="openai/gpt-oss-20b",
    max_new_tokens=512,
    device_map="auto"
)


# ─────────────────────── 2. BRAVE SEARCH TOOL  ────────────────────────
class BraveSearchTool:
    """Wrapper minimale per Brave Search API (richiede BRAVE_API_KEY)."""
    _URL = "https://api.search.brave.com/res/v1/web/search"
    def __init__(self, api_key: str | None = None, top_k: int = 10, timeout: int = 10):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Imposta la variabile d'ambiente BRAVE_API_KEY")
        self.top_k, self.timeout = top_k, timeout

    def run(self, query: str, **_) -> str:
        hdrs = {"X-Subscription-Token": self.api_key}
        params = {"q": query, "count": self.top_k}
        r = requests.get(self._URL, headers=hdrs, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return "\n".join(f"{item['title']} – {item['url']}" for item in data["results"])

# ───────────── 3. TTL CACHE DECORATOR (in-memory, 1 ora) ──────────────
def ttl_cache(ttl: int = 3600, maxsize: int = 512):
    def decorator(fn):
        @functools.lru_cache(maxsize=maxsize)
        def cached(ts_key, *a, **k):
            return fn(*a, **k)
        @functools.wraps(fn)
        def wrapper(*a, **k):
            return cached(round(time.time() / ttl), *a, **k)
        wrapper.cache_clear = cached.cache_clear
        return wrapper
    return decorator

# ──────────────── 4. ROUTER: CACHE, THROTTLE, FALLBACK ────────────────
class SearchRouter:
    """Tenta DuckDuckGo, effettua retry con back-off, poi passa a Brave."""
    def __init__(
        self,
        ddg_tool,
        brave_tool,
        retries: int = 3,
        backoff: int = 2,
        ttl: int = 3600,
    ):
        self.ddg, self.brave = ddg_tool, brave_tool
        self.retries, self.backoff = retries, backoff
        self._cached_run = ttl_cache(ttl)(self._run_uncached)
        self.lock = threading.Lock()

    def run(self, query: str, **kw):
        return self._cached_run(query, **kw)

    def _run_uncached(self, query: str, **kw):
        with self.lock:  # serializza i retry
            for attempt in range(1, self.retries + 1):
                try:
                    return self.ddg.run(query, **kw)
                except Exception as e:
                    wait = self.backoff ** attempt
                    logging.warning(f"DDG errore {attempt}/{self.retries}: {e} – "
                                    f"ritento tra {wait}s")
                    time.sleep(wait)
            logging.info("Troppi errori DDG ⇒ switch a Brave")
            return self.brave.run(query, **kw)

# ───────────────────── 5. ISTANZA DEI TOOLS/AGENT  ─────────────────────
router_tool = SearchRouter(
    ddg_tool=DuckDuckGoSearchTool(),
    brave_tool=BraveSearchTool()
)
agent = CodeAgent(tools=[router_tool], model=model)

# ───────────────────────────── 6. WORKFLOW  ────────────────────────────
with open("lista_biomarkers.txt") as f:
    markers = [ln.strip() for ln in f if ln.strip()]

tag = ".assistantfinal"

with open("biomarkers_with_acronyms.txt", "a") as fout:
    for marker in markers:
        prompt = (f"Is {marker} a valid biomarker? If yes, give its standard acronym. "
                  "If none exists, reply 'N/A'. If not a biomarker, reply 'N/B'. "
                  "Answer only with the acronym, 'N/A', or 'N/B'.")

        try:
            answer = agent.run(prompt, max_steps=3)
            acronym = answer.split(tag, 1)[-1].strip() if tag in answer else "N/A"
        except DuckDuckGoSearchException as e:
            logging.warning(f"DDG rate-limit per {marker}: {e}")
            acronym = "N/A-DDG_limit"           # fallback
        except requests.RequestException as e:
            logging.error(f"Brave error per {marker}: {e}")
            acronym = "N/A-brave"
        except Exception:
            logging.exception("Errore imprevisto")
            acronym = "N/A-general"

        fout.write(f"{marker}: {acronym}\n")  # la riga viene sempre salvata

