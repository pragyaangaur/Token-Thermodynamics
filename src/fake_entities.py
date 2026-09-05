"""Forced-confabulation control set: entities that certainly do not exist.

Any confident answer here is a confabulation by construction, so this condition
separates 'the model is wrong' from 'the model has nothing to be right about'.
"""
import json, numpy as np
rng = np.random.default_rng(7)
ON = ["Kra","Bel","Zor","Mal","Tir","Vas","Dor","Ques","Nar","Pel","Sorn","Xan","Grel","Vund","Ilm","Thas","Orm","Yel","Cad","Rhun"]
MID = ["vo","du","ma","ti","se","ra","no","li","ke","zu","pa","gri","tho","van","el"]
END = ["nia","stan","land","ovia","mark","gard","ria","dor","heim","esia","burg","antia","opol","vok","ador"]
ADJ = ["Hollow","Crimson","Silent","Ninth","Pale","Iron","Glass","Bitter","Vanishing","Amber","Quiet","Restless","Broken","Distant","Hidden"]
NOUN = ["Cartographer","Meridian","Lantern","Aviary","Orchard","Almanac","Ledger","Cipher","Quarry","Harbour","Sextant","Threshold","Bellwether","Foundry","Kestrel"]

def word(): return rng.choice(ON) + (rng.choice(MID) if rng.random() < .6 else "") + rng.choice(END)
def title(): return f"The {rng.choice(ADJ)} {rng.choice(NOUN)}"

out = []
seen = set()
while len([x for x in out if x["rel"] == "capital"]) < 220:
    w = word()
    if w in seen: continue
    seen.add(w); out.append({"rel": "capital", "subject": w, "object": "<<NONEXISTENT>>", "links": 0})
while len([x for x in out if x["rel"] == "element_symbol"]) < 180:
    w = word().replace("nia", "nium") + ("ium" if rng.random() < .5 else "on")
    if w in seen: continue
    seen.add(w); out.append({"rel": "element_symbol", "subject": w, "object": "<<NONEXISTENT>>", "links": 0})
for rel, n in [("director", 300), ("author", 300)]:
    while len([x for x in out if x["rel"] == rel]) < n:
        t = title() + " of " + word() if rng.random() < .4 else title()
        if t in seen: continue
        seen.add(t); out.append({"rel": rel, "subject": t, "object": "<<NONEXISTENT>>", "links": 0})
json.dump(out, open("data/fake_entities.json", "w"))
print(len(out), "fake items"); print([x["subject"] for x in out[:4]], [x["subject"] for x in out[-4:]])
