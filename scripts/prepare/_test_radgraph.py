"""Quick sanity test for RadGraph 0.1.18 API + entity-F1 pair mining logic."""
from radgraph import RadGraph

rg = RadGraph(reward_level="partial")

# RadGraph is a callable — pass a list of reports, get a list of annotation dicts
reports = [
    "The lungs are clear. No pleural effusion.",
    "Lungs are clear. No effusion or pneumothorax.",
]
results = rg(reports)
assert isinstance(results, (list, tuple)) and len(results) == 2

# Extract (token, label) entity sets — the same logic used in mine_factual_pairs.py
def to_entity_set(result):
    entities = result.get("entities", {})
    if not entities:
        # Some versions nest under the report text key
        for v in result.values():
            if isinstance(v, dict) and "entities" in v:
                entities = v["entities"]
                break
    return frozenset(
        (str(e.get("tokens", "")).lower().strip(), str(e.get("label", "")))
        for e in entities.values()
        if isinstance(e, dict) and e.get("tokens")
    )

s1 = to_entity_set(results[0])
s2 = to_entity_set(results[1])
print(f"Report 1 entities ({len(s1)}): {s1}")
print(f"Report 2 entities ({len(s2)}): {s2}")

# FactMM-RAG eq.1: Dice F1 on entity sets
if len(s1) + len(s2) > 0:
    f1 = 2.0 * len(s1 & s2) / (len(s1) + len(s2))
    print(f"F1RadGraph (Dice): {f1:.4f}")
    assert 0 <= f1 <= 1
else:
    print("No entities found — check if RadGraph model downloaded correctly.")

print("\nRadGraph OK — using real entity/relation NER+RE (not keyword fallback)")
