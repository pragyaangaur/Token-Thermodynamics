import json, sys
sys.path.insert(0,'src')
from fetch_data import sparql, V
recs=[]
# stratify by fame band so we get both famous and obscure people
BANDS=[(200,10000,300),(100,200,400),(60,100,400),(40,60,400),(28,40,500)]
for lo,hi,lim in BANDS:
    q=f"""SELECT ?sLabel ?dob ?links WHERE {{
      ?s wdt:P31 wd:Q5 ; wikibase:sitelinks ?links ; wdt:P569 ?dob .
      FILTER(?links >= {lo} && ?links < {hi})
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} LIMIT {lim}"""
    rows=sparql(q)
    n=0
    for r in rows:
        s=V(r,'sLabel'); d=V(r,'dob'); L=V(r,'links')
        if not s or not d or d[0]=='-': continue
        if s.startswith('Q') and s[1:].isdigit(): continue
        recs.append({"subject":s,"object":d[:4],"links":int(L)}); n+=1
    print(f"band {lo}-{hi}: {n}", flush=True)
raw=json.load(open('data/wikidata_raw.json'))
# dedupe by subject, keep first
seen=set(); ded=[]
for r in recs:
    if r['subject'] in seen: continue
    seen.add(r['subject']); ded.append(r)
raw['birth_year']=ded
json.dump(raw, open('data/wikidata_raw.json','w'))
print('birth_year total', len(ded))
print({k:len(v) for k,v in raw.items()})
