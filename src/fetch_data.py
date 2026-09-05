import json, time, urllib.parse, urllib.request, os, sys
EP = "https://query.wikidata.org/sparql"
HDR = {"Accept": "application/sparql-results+json", "User-Agent": "thermo-uncertainty-research/1.0"}

def sparql(q, tries=3):
    for k in range(tries):
        try:
            url = EP + "?" + urllib.parse.urlencode({"query": q})
            with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=180) as r:
                return json.load(r)["results"]["bindings"]
        except Exception as e:
            print("  retry", k, type(e).__name__, str(e)[:120], file=sys.stderr); time.sleep(5)
    return []

def V(row, k): return row[k]["value"] if k in row else None

QUERIES = {
# relation: (sparql, subject key, object key, extra keys)
"capital": """SELECT ?sLabel ?oLabel ?links WHERE {
  ?s wdt:P31 wd:Q3624078 ; wdt:P36 ?o . ?s wikibase:sitelinks ?links .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }""",

"element_symbol": """SELECT ?sLabel ?o ?links WHERE {
  ?s wdt:P31 wd:Q11344 ; wdt:P246 ?o . ?s wikibase:sitelinks ?links .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }""",

"director": """SELECT ?sLabel ?oLabel ?links WHERE {
  ?s wdt:P31 wd:Q11424 ; wdt:P57 ?o ; wikibase:sitelinks ?links .
  FILTER(?links > 12)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }
  ORDER BY DESC(?links) LIMIT 1600""",

"birth_year": """SELECT ?sLabel ?dob ?links WHERE {
  ?s wdt:P31 wd:Q5 ; wdt:P569 ?dob ; wikibase:sitelinks ?links .
  FILTER(?links > 18)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }
  ORDER BY DESC(?links) LIMIT 2000""",

"author": """SELECT ?sLabel ?oLabel ?links WHERE {
  ?s wdt:P31 wd:Q7725634 ; wdt:P50 ?o ; wikibase:sitelinks ?links .
  FILTER(?links > 8)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }
  ORDER BY DESC(?links) LIMIT 1200""",
}

out = {}
for name, q in QUERIES.items():
    print("fetching", name, flush=True)
    rows = sparql(q)
    recs = []
    for r in rows:
        s = V(r, "sLabel"); o = V(r, "oLabel") or V(r, "o"); L = V(r, "links")
        if not s or not o: continue
        if s.startswith("Q") and s[1:].isdigit(): continue
        if o.startswith("Q") and o[1:].isdigit(): continue
        if name == "birth_year":
            o = o[:4] if o[0] != "-" else None
            if not o: continue
        recs.append({"subject": s, "object": o, "links": int(L) if L else 0})
    out[name] = recs
    print(f"  {name}: {len(recs)} rows", flush=True)

os.makedirs("data", exist_ok=True)
json.dump(out, open("data/wikidata_raw.json", "w"))
print("saved data/wikidata_raw.json")
