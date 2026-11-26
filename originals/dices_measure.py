"""Measure how much disagreement in DICES is structured rather than rater noise.

Usage: python3 -I dices_measure.py <dices-dataset-main dir> <output json>
"""
import json
import sys
from itertools import combinations

import numpy as np
import pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]
RNG = np.random.default_rng(20261115)

REMOVED = {
    "350": ['297514565398139', '297515609163939', '297515750682315', '297515617432733',
            '297541515566649', '297541515769980', '297515629971478', '297059995361243',
            '297541522412126', '297540556928761', '297541321453321', '297540562350921',
            '297540983991638', '297060365288109', '297514543980607', '297515729806999',
            '297541271027233', '296709611112092', '296709543131761'],
    "990": ['296768824962581', '296740172100277', '296729424498056', '296835612486158',
            '297212198239077', '296740254381555', '296740049152799', '297212002905763',
            '296740761150184', '296708974347177', '296770194464643', '296768399729683'],
}
REASONS = {
    "350": ["Q2_harmful_content_overall", "Q3_bias_overall", "Q4_misinformation",
            "Q5_political_affiliation", "Q6_policy_guidelines_overall"],
    "990": ["Q2_harmful_content_overall", "Q3_unfair_bias_overall", "Q4_misinformation_overall"],
}
FILES = {"350": "350/diverse_safety_adversarial_dialog_350.csv",
         "990": "990/diverse_safety_adversarial_dialog_990.csv"}
N_SPLITS = 500
N_NULL = 200


def contested(p, thr):
    return np.minimum(p, 1 - p) >= thr


def item_rates(m):
    """m: items x raters matrix with 1 (unsafe), 0 (safe), nan (unsure/missing)."""
    yes = np.nansum(m, axis=1)
    n = np.sum(~np.isnan(m), axis=1)
    return yes / np.where(n == 0, np.nan, n)


def analyse(key):
    df = pd.read_csv(f"{ROOT}/{FILES[key]}", low_memory=False, dtype={"rater_id": str})
    raw_rows = len(df)
    df = df[~df.rater_id.isin(REMOVED[key])].copy()
    res = {"dataset": f"DICES-{key}", "rows_raw": raw_rows, "rows_kept": len(df),
           "items": int(df.item_id.nunique()), "raters": int(df.rater_id.nunique())}
    res["raters_per_item_mean"] = float(df.groupby("item_id").size().mean())
    res["unsure_share"] = float((df.Q_overall == "Unsure").mean())

    lab = df.Q_overall.map({"Yes": 1.0, "No": 0.0})
    mat = (df.assign(y=lab).pivot_table(index="item_id", columns="rater_id", values="y", aggfunc="first"))
    M = mat.to_numpy(dtype=float)
    items, raters = M.shape
    p = item_rates(M)
    res["unsafe_share_overall"] = float(np.nanmean(M))
    for thr in (0.2, 1 / 3):
        res[f"contested_share_min{thr:.2f}"] = float(np.mean(contested(p, thr)))
    res["p_quantiles"] = {q: float(np.nanquantile(p, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}

    # Split-half reproducibility of item-level unsafe shares.
    rs, keep = [], []
    for _ in range(N_SPLITS):
        perm = RNG.permutation(raters)
        a, b = perm[: raters // 2], perm[raters // 2:]
        pa, pb = item_rates(M[:, a]), item_rates(M[:, b])
        ok = ~np.isnan(pa) & ~np.isnan(pb)
        rs.append(np.corrcoef(pa[ok], pb[ok])[0, 1])
        ca, cb = contested(pa, 0.2), contested(pb, 0.2)
        if ca.sum():
            keep.append((ca & cb).sum() / ca.sum())
    r = float(np.mean(rs))
    res["split_half_r_mean"] = r
    res["spearman_brown_reliability"] = float(2 * r / (1 + r))
    res["contested_in_half_A_also_in_half_B"] = float(np.mean(keep))

    # Null model: one true label per item (majority), and each rater errs with
    # its own class-conditional rates (two-coin Dawid-Skene, estimated against
    # the majority). Disagreement in the null is rater error only.
    truth = (p >= 0.5).astype(float)
    obs = ~np.isnan(M)
    sens = np.array([np.nanmean(M[truth == 1, j]) for j in range(raters)])  # P(yes | truth yes)
    fpr = np.array([np.nanmean(M[truth == 0, j]) for j in range(raters)])   # P(yes | truth no)
    sens, fpr = np.nan_to_num(sens, nan=0.5), np.nan_to_num(fpr, nan=0.5)
    null_c20, null_c33 = [], []
    for _ in range(N_NULL):
        prob = np.where(truth[:, None] == 1, sens[None, :], fpr[None, :])
        sim = (RNG.random(M.shape) < prob).astype(float)
        sim[~obs] = np.nan
        ps = item_rates(sim)
        null_c20.append(np.mean(contested(ps, 0.2)))
        null_c33.append(np.mean(contested(ps, 1 / 3)))
    res["null_single_truth_contested_min0.20_mean"] = float(np.mean(null_c20))
    res["null_single_truth_contested_min0.20_max"] = float(np.max(null_c20))
    res["null_single_truth_contested_min0.33_mean"] = float(np.mean(null_c33))
    res["null_single_truth_contested_min0.33_max"] = float(np.max(null_c33))

    # Same label, different grounds: among raters who call an item unsafe,
    # do they cite the same reason category?
    cats = REASONS[key]
    yes = df[df.Q_overall == "Yes"]
    top_lt_half, jacc, n_items = 0, [], 0
    for _, g in yes.groupby("item_id"):
        if len(g) < 10:
            continue
        n_items += 1
        sets = [frozenset(c for c in cats if row[c] == "Yes") for _, row in g[cats].iterrows()]
        counts = {c: sum(c in s for s in sets) for c in cats}
        if max(counts.values()) / len(sets) < 0.5:
            top_lt_half += 1
        pairs = list(combinations(range(len(sets)), 2))
        if len(pairs) > 2000:
            idx = RNG.choice(len(pairs), 2000, replace=False)
            pairs = [pairs[i] for i in idx]
        vals = [len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i, j in pairs if sets[i] | sets[j]]
        if vals:
            jacc.append(np.mean(vals))
    res["reason_items_with_10plus_unsafe_votes"] = n_items
    res["reason_share_items_top_category_below_half"] = top_lt_half / n_items if n_items else None
    res["reason_mean_pairwise_jaccard"] = float(np.mean(jacc)) if jacc else None
    res["reason_share_items_jaccard_below_0.5"] = float(np.mean(np.array(jacc) < 0.5)) if jacc else None
    res["reason_categories"] = cats
    return res


out = [analyse(k) for k in ("350", "990")]
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))


def overdispersion(key):
    """Compare the spread of item-level unsafe shares with the single-truth null."""
    df = pd.read_csv(f"{ROOT}/{FILES[key]}", low_memory=False, dtype={"rater_id": str})
    df = df[~df.rater_id.isin(REMOVED[key])]
    lab = df.Q_overall.map({"Yes": 1.0, "No": 0.0})
    M = df.assign(y=lab).pivot_table(index="item_id", columns="rater_id", values="y", aggfunc="first").to_numpy(dtype=float)
    p = item_rates(M)
    truth = (p >= 0.5).astype(float)
    obs = ~np.isnan(M)
    sens = np.nan_to_num(np.array([np.nanmean(M[truth == 1, j]) if (truth == 1).any() else np.nan for j in range(M.shape[1])]), nan=0.5)
    fpr = np.nan_to_num(np.array([np.nanmean(M[truth == 0, j]) for j in range(M.shape[1])]), nan=0.5)
    vr, low_null = [], []
    for _ in range(N_NULL):
        prob = np.where(truth[:, None] == 1, sens[None, :], fpr[None, :])
        sim = (RNG.random(M.shape) < prob).astype(float)
        sim[~obs] = np.nan
        ps = item_rates(sim)
        vr.append(np.nanvar(ps))
        low_null.append(np.mean(np.minimum(ps, 1 - ps) < 0.1))
    return {"dataset": f"DICES-{key}",
            "var_p_observed": float(np.nanvar(p)), "var_p_null_mean": float(np.mean(vr)),
            "variance_ratio": float(np.nanvar(p) / np.mean(vr)),
            "near_unanimous_share_observed": float(np.mean(np.minimum(p, 1 - p) < 0.1)),
            "near_unanimous_share_null_mean": float(np.mean(low_null))}


if __name__ == "__main__":
    extra = [overdispersion(k) for k in ("350", "990")]
    with open(OUT.replace(".json", "_dispersion.json"), "w") as f:
        json.dump(extra, f, indent=2)
    print(json.dumps(extra, indent=2))
