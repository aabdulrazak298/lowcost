#!/usr/bin/env python3
"""Analyze LowCostLLM cache.db — real usage patterns to drive improvement decisions."""
import sqlite3, re, sys, os
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache.db")
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

def q(sql, *args):
    return c.execute(sql, args).fetchall()

print("=" * 70)
print("SCHEMA — qa_cache")
for r in q("PRAGMA table_info(qa_cache)"):
    print(f"  {r['name']:<20} {r['type']:<12} {'PK' if r['pk'] else ''}")

print("\n" + "=" * 70)
print("OVERALL COUNTS")
total = q("SELECT COUNT(*) n FROM qa_cache")[0]["n"]
print(f"  qa_cache rows:            {total}")
try:
    fts = q("SELECT COUNT(*) n FROM qa_cache_fts")[0]["n"]
    print(f"  qa_cache_fts rows:        {fts}  ({'OK' if fts==total else 'MISMATCH!'})")
except Exception as e:
    print(f"  fts: {e}")

rng = q("SELECT MIN(created_at) lo, MAX(created_at) hi, COUNT(DISTINCT date(created_at)) days FROM qa_cache")[0]
print(f"  created range (UTC):      {rng['lo']}  →  {rng['hi']}  ({rng['days']} distinct days)")

print("\n-- purpose split")
for r in q("SELECT purpose, COUNT(*) n FROM qa_cache GROUP BY purpose"):
    print(f"  {r['purpose']:<8} {r['n']}")

print("\n-- model_used split")
for r in q("SELECT model_used, COUNT(*) n FROM qa_cache GROUP BY model_used ORDER BY n DESC"):
    print(f"  {r['model_used']:<40} {r['n']}")

print("\n-- entries per day (created_at UTC)")
for r in q("SELECT date(created_at) d, COUNT(*) n FROM qa_cache GROUP BY d ORDER BY d"):
    print(f"  {r['d']}  {r['n']}")

print("\n" + "=" * 70)
print("HIT ANALYSIS")
h0 = q("SELECT COUNT(*) n FROM qa_cache WHERE hit_count = 0")[0]["n"]
h1 = q("SELECT COUNT(*) n FROM qa_cache WHERE hit_count >= 1")[0]["n"]
hs = q("SELECT COALESCE(SUM(hit_count),0) n FROM qa_cache")[0]["n"]
print(f"  never hit:  {h0} ({100*h0/max(total,1):.0f}%)")
print(f"  hit >=1:    {h1}")
print(f"  total hits: {hs}")
top = q("SELECT id, hit_count, purpose, substr(query,1,60) q FROM qa_cache ORDER BY hit_count DESC LIMIT 10")
print("\n-- top 10 by hit_count")
for r in top:
    print(f"  hit={r['hit_count']:<4} {r['purpose']:<5} | {r['q']}")

print("\n-- hit_count distribution")
for r in q("SELECT hit_count, COUNT(*) n FROM qa_cache GROUP BY hit_count ORDER BY hit_count"):
    print(f"  {r['hit_count']} hits: {r['n']} entries")

print("\n" + "=" * 70)
print("DUPLICATE / NEAR-DUPLICATE QUERIES (same normalized query stored >1x)")
dups = q("""SELECT LOWER(TRIM(query)) qq, COUNT(*) n, GROUP_CONCAT(id) ids, GROUP_CONCAT(hit_count) hits
            FROM qa_cache GROUP BY qq HAVING n > 1 ORDER BY n DESC LIMIT 15""")
if not dups:
    print("  (none)")
for r in dups:
    print(f"  x{r['n']} hits[{r['hits']}] ids={r['ids']} | {r['qq'][:70]}")

print("\n" + "=" * 70)
print("YOUTUBE VIDEO RE-ASKS (multiple entries containing same video ID)")
vid_re = re.compile(r"(?:v=|youtu\.be/|youtube\.com/embed/|/shorts/|video_url_or_id[\"': ]+)([A-Za-z0-9_-]{11})")
per_vid = defaultdict(list)
for r in q("SELECT id, query, hit_count, created_at, model_used FROM qa_cache"):
    vids = set(vid_re.findall(r["query"]))
    for v in vids:
        per_vid[v].append((r["id"], r["hit_count"], r["created_at"], r["query"][:60]))
multi = {v: x for v, x in per_vid.items() if len(x) > 1}
print(f"  distinct video IDs referenced: {len(per_vid)}, with >1 entry each: {len(multi)}")
for v, x in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:10]:
    print(f"  {v}: {len(x)} entries | ids={[e[0] for e in x]} hits={[e[1] for e in x]} | first: {x[0][3]}")

print("\n" + "=" * 70)
print("URL PATTERNS IN QUERIES")
url_re = re.compile(r"https?://\S+")
n_url = sum(1 for r in q("SELECT query FROM qa_cache") if url_re.search(r["query"]))
print(f"  queries containing a URL: {n_url} ({100*n_url/max(total,1):.0f}%)")
domains = Counter()
for r in q("SELECT query FROM qa_cache"):
    for m in url_re.findall(r["query"]):
        dm = re.sub(r"^https?://(www\.)?", "", m).split("/")[0]
        domains[dm] += 1
print("  top domains:")
for dm, n in domains.most_common(10):
    print(f"    {dm:<40} {n}")

print("\n-- query length distribution (words)")
lens = q("SELECT length(query) - length(replace(query,' ','')) + 1 w FROM qa_cache")
dist = Counter(min((r["w"]//5)*5, 50) for r in lens)
for k in sorted(dist):
    print(f"  {k:>3}-{k+4} words: {dist[k]}")

print("\n" + "=" * 70)
print("RECENT ACTIVITY (last 25 rows, UTC)")
for r in q("SELECT id,purpose,model_used,hit_count,created_at,substr(query,1,65) q FROM qa_cache ORDER BY created_at DESC,id DESC LIMIT 25"):
    print(f"  #{r['id']:<4} {r['purpose']:<5} {r['model_used'][:28]:<28} hit={r['hit_count']:<3} {r['created_at']} | {r['q']}")

print("\n" + "=" * 70)
print("STATS SNAPSHOTS")
for t in ("stats_snapshot", "stats_snapshot_code"):
    try:
        rows = q(f"SELECT * FROM {t}")
        print(f"  {t}: {len(rows)} row(s)")
        for r in rows[:2]:
            for k in r.keys():
                print(f"    {k}: {r[k]}")
    except Exception as e:
        print(f"  {t}: {e}")

print("\n" + "=" * 70)
print("OTHER TABLES")
for r in q("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
    try:
        n = q(f"SELECT COUNT(*) n FROM {r['name']}")[0]["n"]
        print(f"  {r['name']:<25} {n} rows")
    except Exception:
        print(f"  {r['name']:<25} (?)")

c.close()
