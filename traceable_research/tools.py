"""Read-only research tools. These are the only actions an agent can take.

search_papers  ~ web_search : discover public scholarly sources (OpenAlex, no key needed)
fetch_source   ~ web_extract: open the source record and return its full abstract text
verify_doi     : check that a citation resolves to the intended work (Crossref)
"""
import html
import re

import requests

from .guardrails import wrap_untrusted
from .llm import cached

UA = {"User-Agent": "traceable-research-capstone/1.0 (education project)"}


def _get(url, params=None):
    payload = {"url": url, "params": params or {}}

    def fetch():
        resp = requests.get(url, params=params, headers=UA, timeout=30)
        return {"status": resp.status_code, "json": resp.json() if resp.ok else None}

    data, _ = cached("http", payload, fetch)
    return data


def _abstract(inverted):
    if not inverted:
        return ""
    pos = sorted((p, word) for word, places in inverted.items() for p in places)
    return " ".join(word for _, word in pos)


def _record(work):
    abstract = _abstract(work.get("abstract_inverted_index"))
    authors = [a["author"]["display_name"] for a in (work.get("authorships") or [])[:3] if a.get("author")]
    venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    return {
        "source_id": work["id"].rsplit("/", 1)[-1],
        "title": work.get("title") or work.get("display_name") or "",
        "year": work.get("publication_year"),
        "authors": authors,
        "venue": venue,
        "doi": work.get("doi"),
        "cited_by": work.get("cited_by_count", 0),
        "abstract": abstract,
    }


SELECT = "id,doi,title,display_name,publication_year,cited_by_count,abstract_inverted_index,authorships,primary_location"


def search_papers(query, k=5):
    """Search OpenAlex. Returns short snippets only: agents must fetch_source before relying on a source."""
    query = re.sub(r"[^\w\s\-']", " ", query)[:200]
    data = _get("https://api.openalex.org/works", {
        "search": query, "per-page": k, "select": SELECT,
        "filter": "has_abstract:true,language:en,type:article|review",
    })
    if data["status"] != 200 or not data["json"]:
        return {"error": f"search failed with HTTP {data['status']}"}
    results = []
    for work in data["json"]["results"]:
        rec = _record(work)
        results.append({k2: rec[k2] for k2 in ("source_id", "title", "year", "doi", "cited_by")}
                       | {"snippet": rec["abstract"][:240]})
    return {"query": query, "results": results}


def fetch_source(source_id):
    """Open one OpenAlex work by id (W123...) and return its metadata and full abstract (untrusted text)."""
    source_id = str(source_id).strip().rsplit("/", 1)[-1]
    if not re.fullmatch(r"W\d+", source_id):
        return {"error": f"'{source_id}' is not an OpenAlex work id like W2052218930"}
    data = _get(f"https://api.openalex.org/works/{source_id}", {"select": SELECT})
    if data["status"] != 200 or not data["json"]:
        return {"error": f"source {source_id} not found (HTTP {data['status']})"}
    rec = _record(data["json"])
    rec["abstract"], rec["injection_flags"] = wrap_untrusted(rec["abstract"])
    return rec


def _same_title(a, b):
    norm = lambda t: re.sub(r"[^a-z0-9]", "", html.unescape(t or "").lower())
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    short, long_ = sorted((a, b), key=len)
    # registries sometimes store only the main title before the colon: accept a clear prefix match
    return a[:60] == b[:60] or (len(short) >= 12 and long_.startswith(short))


def verify_doi(doi, expected_title):
    """Citation check: does the DOI resolve to a work with the expected title? Crossref first, then DataCite
    (university repositories and theses often register DOIs with DataCite, which Crossref does not know)."""
    if not doi:
        return {"resolves": False, "title_match": False, "note": "no DOI"}
    bare = doi.replace("https://doi.org/", "")
    data = _get(f"https://api.crossref.org/works/{bare}")
    if data["status"] == 200 and data["json"]:
        title = (data["json"]["message"].get("title") or [""])[0]
        return {"resolves": True, "title_match": _same_title(title, expected_title), "registry": "crossref",
                "registry_title": title}
    data = _get(f"https://api.datacite.org/dois/{bare}")
    if data["status"] == 200 and data["json"]:
        titles = data["json"]["data"]["attributes"].get("titles") or [{}]
        title = titles[0].get("title", "")
        return {"resolves": True, "title_match": _same_title(title, expected_title), "registry": "datacite",
                "registry_title": title}
    return {"resolves": False, "title_match": False, "note": f"not found in Crossref or DataCite"}


TOOLS = {
    "search_papers": {"fn": search_papers, "args": {"query": "str"},
                      "desc": "Search public scholarly literature. Returns ids, titles, years and short snippets."},
    "fetch_source": {"fn": fetch_source, "args": {"source_id": "str"},
                     "desc": "Open a source by its id (e.g. W2052218930) and read its full abstract."},
}
