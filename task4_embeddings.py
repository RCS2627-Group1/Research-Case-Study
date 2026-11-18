import re
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer

# ---------- load data ----------
df = pd.read_csv("dog_whistle_reddit_sample_5000.csv", encoding="utf-8-sig")
gloss = pd.read_csv("silent_signals_reference_dictionary.csv", encoding="utf-8-sig")

TEXT = "content"
df[TEXT] = df[TEXT].fillna("").astype(str)
df["dog_whistle"] = df["dog_whistle"].fillna("").astype(str).str.lower().str.strip()

glossary = set(gloss["dog_whistle"].dropna().str.lower().str.strip()) | \
           set(gloss["dog_whistle_root"].dropna().str.lower().str.strip())

# ---------- choose seed terms automatically ----------
counts = df[df["dog_whistle"] != ""]["dog_whistle"].value_counts()
seed_terms = list(counts[counts >= 20].index[:5])
print("Seed terms:", seed_terms)
print(counts.head(10))

# ---------- embed all posts ----------
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(df[TEXT].tolist(), batch_size=64,
                   normalize_embeddings=True, show_progress_bar=True)

# ---------- similarity to each seed profile ----------
lower = df[TEXT].str.lower()
rows = []
for t in seed_terms:
    seed_idx = np.where(df["dog_whistle"] == t)[0]
    # pool = posts that do not contain this term at all
    pool = np.where((df["dog_whistle"] != t) & (~lower.str.contains(re.escape(t))))[0]
    centroid = emb[seed_idx].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    sims = emb[pool] @ centroid
    for j in np.argsort(-sims)[:150]:
        rows.append((t, int(pool[j]), float(sims[j]), df[TEXT].iloc[pool[j]]))

sim_df = pd.DataFrame(rows, columns=["seed_term", "post_id", "similarity", "post"])
sim_df["rank"] = sim_df.groupby("seed_term")["similarity"].rank(ascending=False, method="first").astype(int)
sim_df.to_csv("task4_similar_posts.csv", index=False, encoding="utf-8-sig")

# ---------- candidate expressions from similar posts ----------
STOP = set("""a an the and or but of to in on at for with from by as is are was were be been
i you he she it we they me my your his her its our their this that these those not no so if
then than too very can will just do does did have has had im dont its i'm it's that's""".split())

def ngrams(text, n):
    tok = re.findall(r"[a-z0-9']+", text.lower())
    return [" ".join(tok[i:i+n]) for i in range(len(tok) - n + 1)]

def not_all_stop(e):
    return not all(w in STOP for w in e.split())

sim_texts = sim_df["post"].drop_duplicates().tolist()
top, base = Counter(), Counter()
for n in (1, 2, 3):
    for x in sim_texts:
        top.update(set(ngrams(x, n)))      # count once per post
    for x in df[TEXT]:
        base.update(set(ngrams(x, n)))

cands = []
for e, c in top.items():
    if c >= 5 and not_all_stop(e) and e not in glossary:
        enrich = (c / len(sim_texts)) / ((base[e] + 1) / len(df))
        cands.append((e, len(e.split()), c, base[e], enrich))

cand_df = pd.DataFrame(cands, columns=["expression", "n", "count_in_similar", "count_overall", "enrichment"])
cand_df = cand_df.sort_values("enrichment", ascending=False).reset_index(drop=True)
cand_df.insert(0, "rank", cand_df.index + 1)
cand_df["in_glossary"] = "no"

# add one example post per candidate
def example(e):
    for x in sim_texts:
        if e in " ".join(re.findall(r"[a-z0-9']+", x.lower())):
            return x[:300]
    return ""
cand_df["example_post"] = cand_df["expression"].apply(example)

cand_df.to_csv("task4_candidate_expressions.csv", index=False, encoding="utf-8-sig")
print("Done. Wrote task4_similar_posts.csv and task4_candidate_expressions.csv")