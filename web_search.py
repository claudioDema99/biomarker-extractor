"""
Script: biomarkers_check_no_agent.py
Requisiti:
    pip install smolagents[transformers] duckduckgo-search requests python-dotenv
    # BRAVE_API_KEY nello .env o come variabile ambiente
"""
from __future__ import annotations
import os, time, functools, logging, threading, requests, torch
from dotenv import load_dotenv
from duckduckgo_search import DDGS, exceptions as ddg_exc
from smolagents import TransformersModel            # LLM locale senza agent
load_dotenv()

# ─────────────── 1. LLM locale (TransformersModel) ────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
llm = TransformersModel(
    model_id="openai/gpt-oss-20b",                  # cambia con il tuo
    max_new_tokens=256,
    device_map="auto"                               # oppure "cpu"
)

# ─────────────── 2. Brave wrapper ────────────────────────────────
class BraveSearch:
    _URL = "https://api.search.brave.com/res/v1/web/search"
    def __init__(self, top_k=10, timeout=10):
        key = os.getenv("BRAVE_API_KEY")
        if not key:
            raise RuntimeError("BRAVE_API_KEY mancante")
        self.hdrs = {"X-Subscription-Token": key}
        self.top_k, self.timeout = top_k, timeout
    def __call__(self, query: str) -> str:
        r = requests.get(
            self._URL,
            headers=self.hdrs,
            params={"q": query, "count": self.top_k},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return "\n".join(f"{d['title']} – {d['url']}" for d in r.json()["results"])

# ─────────────── 3. Cache + back-off router (DDG ⇒ Brave) ─────────
def ttl_cache(ttl=3600, maxsize=512):
    def deco(fn):
        @functools.lru_cache(maxsize=maxsize)
        def cached(t,*a,**k): return fn(*a,**k)
        return lambda *a,**k: cached(round(time.time()/ttl),*a,**k)
    return deco

class WebSearch:
    def __init__(self, retries=3, backoff=2, ttl=3600):
        self.ddg   = DDGS()
        self.brave = BraveSearch()
        self.retries, self.backoff = retries, backoff
        self.lock = threading.Lock()
        self.cached = ttl_cache(ttl)(self._search)

    def __call__(self, query:str)->str: return self.cached(query)

    def _search(self, query:str)->str:
        with self.lock:
            for i in range(1, self.retries+1):
                try:
                    res = self.ddg.text(query, max_results=10)
                    return "\n".join(f"{r['title']} – {r['href']}" for r in res)
                except Exception as e:
                    wait = self.backoff**i
                    logging.warning(f"DDG errore {i}/{self.retries}: {e} – retry {wait}s")
                    time.sleep(wait)
            logging.info("Switch a Brave")
            return self.brave(query)

search_web = WebSearch()

# ─────────────── 4. Helper per interrogare il modello ─────────────
def ask_llm(marker: str, snippets: str, llm) -> str:
    system_msg = (
        "You are a clinical biologist. "
        "Given the web snippets, answer ONLY with:\n"
        "  - the upper-case acronym if the marker is a biomarker;\n"
        "  - 'N/A' if it is a biomarker without acronym;\n"
        "  - 'N/B' if it is NOT a biomarker."
    )
    user_msg = f"Marker: {marker}\n\nWeb snippets:\n{snippets}"
    print(f"{marker}: {snippets}")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_msg}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_msg}]
        },
    ]

    raw_text = llm(messages)          # restituisce già una stringa
    return raw_text.content.split("assistantfinal", 1)[-1].strip()



# ─────────────── 5. Workflow principale ──────────────────────────
with open("lista_biomarkers.txt") as f:
    markers = [m.strip() for m in f if m.strip()]

with open("biomarkers_with_acronyms.txt", "a") as fout:
    for m in markers:
        try:
            snip = search_web(f"{m} biomarker acronym")
            acr = ask_llm(m, snip, llm) 
        except (ddg_exc.DuckDuckGoSearchException, requests.RequestException) as e:
            logging.warning(f"Web search error {m}: {e}")
            acr = "N/A-web"
        except Exception:
            logging.exception(f"LLM error su {m}")
            acr = "N/A-llm"
        fout.write(f"{m}: {acr}\n")
