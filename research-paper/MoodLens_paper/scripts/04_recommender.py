"""Quantitative checks on the mood -> menu recommender. Run from SmashGuys-MoodMenu with its venv."""
import json, os, random, statistics as st, sys, time, itertools
sys.path.insert(0, os.getcwd())
from menu_app.menu import MENU, MOODS, recommend, recommend_table, COURSES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "recommender.json")
rnd = random.Random(11); by_id = {m["id"]: m for m in MENU}
res = {"menu": {"items": len(MENU), "veg": sum(m["veg"] for m in MENU), "chef": sum(bool(m.get("chef")) for m in MENU), "new": sum(bool(m.get("new")) for m in MENU),
                "spicy": sum(bool(m.get("spicy")) for m in MENU),
                "by_course": {c: sum(m["course"] == c for m in MENU) for c in COURSES}, "sections": len({m["section"] for m in MENU}),
                "price_min": min(m["price"] for m in MENU), "price_max": max(m["price"] for m in MENU),
                "tags": len({t for m in MENU for t in m["tags"]}), "allergen_items": {a: sum(a in m["allergens"] for m in MENU) for a in ("egg", "nuts", "mushroom", "alcohol")}}}
moods = list(MOODS)

# ---- 1. constraint compliance over random filter settings ----
N = 3000; viol = {"diet": 0, "spice": 0, "allergen": 0, "budget": 0}; empty = 0; sizes = []; lat = []
for _ in range(N):
    mood = rnd.choice(moods); diet = rnd.choice(["any", "veg", "nonveg"]); spice = rnd.choice([0, 1, 2])
    avoid = rnd.sample(["egg", "nuts", "mushroom", "alcohol"], rnd.choice([0, 0, 1, 2])); budget = rnd.choice([None, None, rnd.randrange(400, 2500, 50)])
    t0 = time.perf_counter(); r = recommend(mood, diet=diet, budget=budget, avoid=avoid, max_spice=spice, seed=rnd.randrange(10**6), hour=rnd.randrange(24)); lat.append((time.perf_counter() - t0) * 1e3)
    items = r["combo"]; sizes.append(len(items)); empty += not items
    for it in items:
        if diet == "veg" and not it["veg"]: viol["diet"] += 1
        if diet == "nonveg" and it["veg"] and it["course"] in ("main", "side"): viol["diet"] += 1
        if it["spicy"] > spice: viol["spice"] += 1
        if set(avoid) & set(it["allergens"]): viol["allergen"] += 1
    if budget and sum(i["price"] for i in items) > budget: viol["budget"] += 1
res["compliance"] = {"trials": N, "violations": viol, "empty_combos": empty, "mean_items_per_combo": st.mean(sizes), "latency_ms_mean": st.mean(lat), "latency_ms_p95": sorted(lat)[int(.95 * N)]}
print("1", res["compliance"], flush=True)

# ---- 2. mood fit: share of combo items carrying a tag the mood actively seeks, vs random picks ----
def seek(mood): return {t for t, w in MOODS[mood]["weights"].items() if w >= 1.5}
def avoid_tags(mood): return {t for t, w in MOODS[mood]["weights"].items() if w <= -1}
fit = {}
for mood in moods:
    s = seek(mood); a = avoid_tags(mood); hit_r = hit_b = bad_r = bad_b = n_r = n_b = 0
    for seed in range(200):
        for it in recommend(mood, seed=seed, hour=19)["combo"]:
            it = by_id[it["id"]]; tags = set(it["tags"]) | ({"chef"} if it.get("chef") else set()) | ({"new"} if it.get("new") else set())
            hit_r += bool(tags & s); bad_r += bool(tags & a); n_r += 1
        for c in COURSES:
            it = rnd.choice([m for m in MENU if m["course"] == c]); tags = set(it["tags"]) | ({"chef"} if it.get("chef") else set()) | ({"new"} if it.get("new") else set())
            hit_b += bool(tags & s); bad_b += bool(tags & a); n_b += 1
    fit[mood] = {"seek_hit_recommender": hit_r / n_r, "seek_hit_random": hit_b / n_b, "avoid_hit_recommender": bad_r / n_r, "avoid_hit_random": bad_b / n_b}
res["mood_fit"] = fit; print("2", {m: (round(v["seek_hit_recommender"], 2), round(v["seek_hit_random"], 2)) for m, v in fit.items()}, flush=True)

# ---- 3. variety and Shuffle ----
div = {}; shuf = []
for mood in moods:
    combos = {tuple(i["id"] for i in recommend(mood, seed=s, hour=19)["combo"]) for s in range(100)}; div[mood] = len(combos)
    for s in range(100):
        a = recommend(mood, seed=s, hour=19); ids = [i["id"] for i in a["combo"]]
        b = recommend(mood, seed=s + 5000, hour=19, away=ids); shuf.append(len(set(ids) - {i["id"] for i in b["combo"]}) / max(1, len(ids)))
res["variety"] = {"distinct_combos_per_100_seeds": div, "shuffle_share_of_dishes_changed": st.mean(shuf), "shuffle_fully_new_pct": sum(x == 1 for x in shuf) / len(shuf) * 100}
print("3", res["variety"]["distinct_combos_per_100_seeds"], round(res["variety"]["shuffle_share_of_dishes_changed"], 3), flush=True)

# ---- 4. learning from Mood Lift: does feedback move a mid-ranked main into (or a top main out of) the combo? ----
def main_rank(mood, k=6):
    picked, order = [], []
    for _ in range(k):
        r = recommend(mood, seed=0, hour=19, courses=("main",), exclude=picked)["combo"]
        if not r: break
        order.append(r[0]["id"]); picked.append(r[0]["id"])
    return order
def share(mood, dish, lift, seeds=300):
    return sum(any(i["id"] == dish for i in recommend(mood, seed=s, hour=19, lift=lift)["combo"]) for s in range(seeds)) / seeds * 100
learn = []
for mood in moods:
    order = main_rank(mood)
    for rank in (3, 4, 5):
        d = order[rank - 1]
        learn.append({"mood": mood, "dish": by_id[d]["name"], "rank_before": rank, "kind": "lift +60 (5 checks)", "pct_before": share(mood, d, {}), "pct_after": share(mood, d, {d: {"avg": 60.0, "n": 5}})})
    d = order[0]
    learn.append({"mood": mood, "dish": by_id[d]["name"], "rank_before": 1, "kind": "lift -40 (5 checks)", "pct_before": share(mood, d, {}), "pct_after": share(mood, d, {d: {"avg": -40.0, "n": 5}})})
res["lift_learning"] = learn
pos = [x for x in learn if x["kind"].startswith("lift +")]; neg = [x for x in learn if x["kind"].startswith("lift -")]
res["lift_learning_summary"] = {"positive_cases": len(pos), "mean_pct_before": st.mean(x["pct_before"] for x in pos), "mean_pct_after": st.mean(x["pct_after"] for x in pos),
                                "negative_cases": len(neg), "neg_mean_pct_before": st.mean(x["pct_before"] for x in neg), "neg_mean_pct_after": st.mean(x["pct_after"] for x in neg)}
print("4", res["lift_learning_summary"], flush=True)

# ---- 5. table mode ----
tab = []
for _ in range(500):
    n = rnd.randint(2, 6); ms = [rnd.choice(moods) for _ in range(n)]; t = recommend_table(ms, seed=rnd.randrange(10**5), hour=19)
    mains = [i["id"] for p in t["people"] for i in p["items"] if i["course"] == "main"]
    tab.append({"people": n, "unique_mains": len(set(mains)) == len(mains), "total_ok": t["total"] == sum(p["subtotal"] for p in t["people"]) + sum(s["price"] * s["qty"] for s in t["shared"])})
res["table"] = {"trials": len(tab), "unique_mains_pct": sum(x["unique_mains"] for x in tab) / len(tab) * 100, "total_consistent_pct": sum(x["total_ok"] for x in tab) / len(tab) * 100}
print("5", res["table"])
json.dump(res, open(OUT, "w"), indent=1, default=float)
