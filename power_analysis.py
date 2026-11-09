"""What could this study have detected, and what did it merely fail to?

`significance.py` reports which of the ranking claims clear p = 0.05 at n = 33 and which do
not. That is half the question. A non-significant rho of 0.306 and a non-significant Δrho of
-0.062 are only the same kind of statement if the design could have found them, so this
measures, for the same candidate count and the same score distributions, (1) the power the
study had against the rho it actually got, (2) the smallest true rho it had an 80% chance of
calling significant, and (3) the candidate count that would resolve the two differences it
currently reports as unresolvable.

Scores are simulated from the *observed marginals*, not a smooth distribution: each row is the
real score multiset (n = 33) or a bootstrap sample of it (n > 33, drawn once and shared by both
scorers and the labels so a candidate stays one candidate), placed onto the ranks of a Gaussian
latent whose correlation with the label latent is read off a calibration grid. The tie
structure is therefore carried over - the dense LLM emits 4 distinct scores across 33
candidates - and so is its consequence: a coarse scorer has a *ceiling* on the rho it can
express at all, which this file prints instead of papering over.

    python power_analysis.py                 # ~15 s, reads the committed per_resume CSVs
    python power_analysis.py --sims 400      # quicker, coarser grid interpolation
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from significance import SEED, _rho_rows, bootstrap_ci, rank_metrics

ALPHA = 0.05
TARGET_POWER = 0.80
NS = (33, 50, 66, 100, 150, 250)
LAT_GRID = np.linspace(0.0, 0.99, 10)   # calibration grid; the rho map is monotone along it

RUNS = {
    "rag, tfidf (LLM)": "per_resume_v2.csv",
    "rag, dense (LLM)": "per_resume_v2_dense.csv",
    "rag, tfidf (no LLM)": "per_resume_v2_fallback.csv",
    "rag, dense (no LLM)": "per_resume_v2_fallback_dense.csv",
}


def assign(z, pool_sorted, rng):
    """Place a marginal onto the ranks of latent z, row by row, preserving that marginal.

    At n = len(pool) every row gets the pool itself, so the simulated scorer has exactly the
    real scorer's tie structure. At other n each row is a bootstrap sample of the pool, which
    preserves the score *distribution* but not the exact counts - the only approximation here,
    and it touches only the n-sweep rows.
    """
    draws, n = z.shape
    ranks = np.argsort(np.argsort(z, axis=1), axis=1)
    if n == len(pool_sorted):
        vals = np.tile(pool_sorted, (draws, 1))
    else:
        idx = rng.integers(0, len(pool_sorted), size=(draws, n))
        vals = np.sort(pool_sorted[idx], axis=1)
    return np.take_along_axis(vals, ranks, axis=1)


def sim_rho(rho_lat, score_pool, label_pool, draws, rng):
    """Spearman rho between two marginals joined by a Gaussian latent correlation."""
    n = len(label_pool)
    z1 = rng.standard_normal((draws, n))
    z2 = rho_lat * z1 + np.sqrt(max(0.0, 1 - rho_lat ** 2)) * rng.standard_normal((draws, n))
    return _rho_rows(assign(z1, score_pool, rng), assign(z2, label_pool, rng))


def curve(score_pool, label_pool, sims, rng, thr=None):
    """Simulated rho, spread and rejection rate along a grid of latent correlations.

    Ties attenuate - latent 0.80 through a 4-value score marginal does not give rho 0.80 - so
    the target-to-latent map is read off this grid instead of being assumed from the copula
    identity. `ceiling` is the best rho this multiset can express even with a near-perfect
    latent, because the top grid row is as close to comonotone as the model gets.
    """
    c = {"latent": LAT_GRID, "mean": [], "sd": [], "power": []}
    for g in LAT_GRID:
        r = sim_rho(g, score_pool, label_pool, sims, rng)
        c["mean"].append(float(np.nanmean(r)))
        c["sd"].append(float(np.nanstd(r)))
        c["power"].append(float(np.nanmean(np.abs(r) >= thr)) if thr else float("nan"))
    for k in ("mean", "sd", "power"):
        c[k] = np.array(c[k])
    c["ceiling"] = float(c["mean"][-1])
    return c


def at(c, target_rho):
    """Interpolate a target rho back along the grid, and say whether it was reachable.

    The grid is the measurement, so latent, power and spread are read off it by interpolation
    rather than assumed from the copula identity. `clamped` is the case the reader needs: a
    target above this scorer's ceiling cannot be simulated, and the top of the grid is returned.
    """
    mono = np.maximum.accumulate(c["mean"])
    clamped = bool(target_rho > c["ceiling"] + 0.005)
    t = float(np.clip(target_rho, mono[0], mono[-1]))
    latent = float(np.interp(t, mono, c["latent"]))
    return {"latent": latent,
            "mean": float(np.interp(latent, c["latent"], c["mean"])),
            "sd": float(np.interp(latent, c["latent"], c["sd"])),
            "power": float(np.interp(latent, c["latent"], c["power"])),
            "err": float(np.interp(latent, c["latent"], c["mean"]) - target_rho),
            "clamped": clamped}


def null_threshold(score_pool, label_pool, draws, rng):
    """|rho| that would beat the permutation null at this n and tie structure."""
    n = len(label_pool)
    z1 = rng.standard_normal((draws, n))
    z2 = rng.standard_normal((draws, n))
    r = np.abs(_rho_rows(assign(z1, score_pool, rng), assign(z2, label_pool, rng)))
    return float(np.nanquantile(r, 1 - ALPHA))


def paired_delta(lat_ay, lat_by, lat_ab, pool_a, pool_b, label_pool, draws, rng):
    """Δrho of two scorers against shared labels, with their observed mutual dependence.

    Column order is [scorer a, scorer b, label], so the matrix entries are a-b, a-label and
    b-label in that layout. Getting this wrong is not caught by numpy: an asymmetric C is
    factorised from its lower triangle alone, so the generator silently produces a different
    dependence structure than the one asked for. The realised matrix is therefore checked.
    """
    n = len(label_pool)
    C = np.array([[1.0, lat_ab, lat_ay], [lat_ab, 1.0, lat_by], [lat_ay, lat_by, 1.0]])
    w, v = np.linalg.eigh(C)
    clipped = bool(w.min() < 1e-8)
    if clipped:                      # the three correlations cannot all hold at once
        C = (v * np.clip(w, 1e-8, None)) @ v.T
    L = np.linalg.cholesky(C)
    assert np.abs(L @ L.T - C).max() < 1e-9, "Cholesky did not reproduce the dependence asked for"
    z = np.einsum("ij,djn->din", L, rng.standard_normal((draws, 3, n)))
    a = assign(z[:, 0], pool_a, rng)
    b = assign(z[:, 1], pool_b, rng)
    y = assign(z[:, 2], label_pool, rng)
    return _rho_rows(a, y) - _rho_rows(b, y), clipped


def marginal_rank_corr(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def paired_cell(sa33, sb33, lab33, n, rho_a, rho_b, sims, rng):
    """One candidate count, null and alternative from the same generator.

    The two target rhos are the ones measured on the real 33 candidates and are held fixed
    across the sweep: an earlier version re-measured them on each bootstrap resample, which
    changed the effect size at every row and made the power column jump between 7% and 88%
    while measuring nothing. One index vector resamples scores and labels together, so at
    n > 33 a resampled candidate keeps its score and its label.
    """
    idx = np.arange(len(lab33)) if n == len(lab33) else rng.integers(0, len(lab33), size=n)
    pa, pb, lp = np.sort(sa33[idx]), np.sort(sb33[idx]), np.sort(lab33[idx])
    lat_ab = marginal_rank_corr(sa33, sb33)
    ca, cb = curve(pa, lp, max(sims // 2, 200), rng), curve(pb, lp, max(sims // 2, 200), rng)
    ta, tb = at(ca, rho_a), at(cb, rho_b)
    # H0 is "both scorers rank equally well", at a rho each can actually express; asking the
    # coarser one for a rho above its ceiling makes the null an artefact of ties.
    mid = min(0.5 * (rho_a + rho_b), ca["ceiling"], cb["ceiling"])
    null, clip_null = paired_delta(at(ca, mid)["latent"], at(cb, mid)["latent"],
                                   lat_ab, pa, pb, lp, sims, rng)
    thr = (float(np.nanquantile(null, ALPHA / 2)), float(np.nanquantile(null, 1 - ALPHA / 2)))
    obs, clip_alt = paired_delta(ta["latent"], tb["latent"], lat_ab, pa, pb, lp, sims, rng)
    return {"n": n, "rho_a": rho_a, "rho_b": rho_b, "lat_ab": lat_ab,
            "power": float(np.nanmean((obs < thr[0]) | (obs > thr[1]))), "thr": thr,
            "model_delta": float(np.nanmean(obs)), "true_delta": rho_a - rho_b,
            "null_delta": float(np.nanmean(null)),
            "achieved": (ta["mean"], tb["mean"]),
            "ceilings": (ca["ceiling"], cb["ceiling"]),
            "reachable": (not ta["clamped"], not tb["clamped"]),
            "psd_clipped": bool(clip_null or clip_alt),
            "distinct": (len(set(pa.tolist())), len(set(pb.tolist())))}


def load_rag(path):
    d = pd.read_csv(path)
    d = d[d["mode"] == "rag"].sort_values("resume_id")
    return d["score"].to_numpy(float), d["human"].to_numpy(float)


def selfcheck(rng, label_pool):
    """assign() must not damage the marginal, the grid must respond to its input, and the
    paired generator must produce no difference when asked for none."""
    pool = np.sort(np.array([80.0] * 19 + [60.0] * 10 + [40.0, 20.0, 10.0, 5.0]))
    rows = assign(rng.standard_normal((40, len(pool))), pool, rng)
    bad = sum(1 for r in rows if sorted(r) != sorted(pool))
    c = curve(pool, label_pool, 800, rng)
    null_d, _ = paired_delta(0.6, 0.6, 0.5, pool, pool, label_pool, 1500, rng)
    ok = (bad == 0 and abs(c["mean"][0]) < 0.03
          and c["mean"][1] < c["mean"][4] < c["mean"][7] < c["ceiling"]
          and abs(at(c, 0.60)["mean"] - 0.60) < 0.04
          and abs(float(np.nanmean(null_d))) < 0.05)
    print(f"self-check: {len(rows)} rows of a {len(pool)}-value pool reproduced exactly, "
          f"{bad} damaged; independent latents give rho {c['mean'][0]:+.3f}; the grid runs "
          f"{c['mean'][1]:+.3f} / {c['mean'][4]:+.3f} / {c['mean'][7]:+.3f} at latent "
          f"{LAT_GRID[1]:.2f} / {LAT_GRID[4]:.2f} / {LAT_GRID[7]:.2f}; target 0.60 lands on "
          f"{at(c, 0.60)['mean']:+.3f}; two scorers asked for the same rho differ by "
          f"{float(np.nanmean(null_d)):+.3f}")
    if not ok:
        raise SystemExit("simulator failed its own self-check")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--data", default=".")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    root = Path(args.data)
    sims = args.sims

    runs = {k: load_rag(root / v) for k, v in RUNS.items()}
    labels = list(runs.values())[0][1]
    observed = {k: float(rank_metrics(s, h)[0]) for k, (s, h) in runs.items()}

    print(f"seed={SEED} sims={sims} per cell | score and label marginals read from the "
          f"committed runs | alpha={ALPHA}, target power={TARGET_POWER}")
    print("rejection rule = the permutation null of the same marginals, so the ties that make "
          "these scorers coarse sit in both the null and the alternative\n")
    selfcheck(rng, np.sort(labels))

    print("distinct scores, and the rho ceiling each score multiset can express")
    print(f"   {'run':<20} {'rho':>7} {'distinct':>9} {'modal share':>12} {'ceiling':>9}")
    for k, (s, h) in runs.items():
        c = curve(np.sort(s), np.sort(h), max(sims // 2, 400), rng)
        print(f"   {k:<20} {observed[k]:>+7.3f} {len(set(s.tolist())):>9} "
              f"{pd.Series(s).value_counts(normalize=True).max():>12.3f} "
              f"{c['ceiling']:>9.3f}")
    print("   the ceiling is what a perfect latent cannot beat: a scorer that emits 4 values")
    print("   across 33 candidates cannot express a perfect ranking of them, and no amount of")
    print("   recalibration or candidates buys it back.")
    print()

    print("1) power the study had against the rho it found, and the model's two checks")
    print(f"   {'run':<20} {'rho':>7} {'null |rho|':>11} {'model rho':>10} {'power':>7} "
          f"{'sim SD':>8} {'bootstrap SD':>13}")
    for k, (s, h) in runs.items():
        pool, lp = np.sort(s), np.sort(h)
        thr = null_threshold(pool, lp, max(sims, 2000), rng)
        a = at(curve(pool, lp, sims, rng, thr), observed[k])
        boot, _ = bootstrap_ci(s, h, max(sims, 2000), rng)
        bs = (boot[1] - boot[0]) / (2 * 1.96)
        flag = "  <- above this scorer's ceiling" if a["clamped"] else ""
        print(f"   {k:<20} {observed[k]:>+7.3f} {thr:>11.3f} {a['mean']:>+10.3f} "
              f"{a['power']:>6.1%} {a['sd']:>8.3f} {bs:>13.3f}{flag}")
    print("   model rho against observed rho is the calibration check, sim SD against")
    print("   bootstrap SD is the width check. Close, but not the same object: the bootstrap")
    print("   resamples this one dataset, the simulation resamples the design.")
    print()

    print("2) smallest true rho detectable at 80% power, by candidate count")
    print(f"   {'n':>4} " + " ".join(f"{k.split(', ')[1]:>18}" for k in runs))
    for n in (33, 66, 100, 200):
        cells = []
        for k, (s, _) in runs.items():
            idx = np.arange(len(s)) if n == len(s) else rng.integers(0, len(s), size=n)
            pool, lp = np.sort(s[idx]), np.sort(labels[idx])
            thr = null_threshold(pool, lp, max(sims, 2000), rng)
            c = curve(pool, lp, max(sims // 2, 400), rng, thr)
            cells.append(float(np.interp(TARGET_POWER, c["power"], c["mean"])))
        print(f"   {n:>4} " + " ".join(f"{cv:>18.3f}" for cv in cells))
    print("   a design that needs rho 0.46 to be significant cannot turn 'rho 0.31, p = 0.09'")
    print("   into a finding about hiring; it is a finding about this n.")
    print()

    print("3) what would resolve the two differences the study reports as unresolvable")
    for a_name, b_name in [("rag, tfidf (LLM)", "rag, dense (LLM)"),
                           ("rag, dense (LLM)", "rag, dense (no LLM)")]:
        sa, ha = runs[a_name]
        sb, _ = runs[b_name]
        print(f"\n   {a_name} vs {b_name}: delta rho {observed[a_name] - observed[b_name]:+.3f} "
              f"held fixed across the sweep, at rho "
              f"{observed[a_name]:+.3f} / {observed[b_name]:+.3f}")
        print(f"     {'n':>4} {'power':>8} {'reject if delta outside':>23} "
              f"{'sim rho a/b':>15} {'model delta':>12} {'true delta':>11} "
              f"{'null delta':>11} {'distinct a/b':>13}")
        cells = [paired_cell(sa, sb, ha, n, observed[a_name], observed[b_name],
                             max(sims // 2, 400), rng) for n in NS]
        for r in cells:
            notes = []
            if not all(r["reachable"]):
                notes.append("target above a ceiling")
            if r["psd_clipped"]:
                notes.append("correlation matrix clipped")
            print(f"     {r['n']:>4} {r['power']:>7.1%} [{r['thr'][0]:+.3f}, {r['thr'][1]:+.3f}]"
                  f"{r['achieved'][0]:>8.3f}/{r['achieved'][1]:<6.3f}"
                  f"{r['model_delta']:>+12.3f}{r['true_delta']:>+11.3f}"
                  f"{r['null_delta']:>+11.3f} {r['distinct'][0]:>6}/{r['distinct'][1]:<6}"
                  f"{'   <- the study as run' if r['n'] == 33 else ''} {' '.join(notes)}")
        worst_null = max(abs(r["null_delta"]) for r in cells)
        rise = all(cells[i]["power"] <= cells[i + 1]["power"] + 0.04 for i in range(len(cells) - 1))
        print(f"     power should not fall as candidates are added: "
              f"{'it rises across this sweep' if rise else 'IT FALLS SOMEWHERE - do not quote this sweep'}")
        print(f"     the null column is the generator's own calibration: under H0 the two "
              f"scorers are asked for the same rho, so its mean delta must sit at 0 "
              f"(worst here {worst_null:+.3f}). A systematic offset means the columns are not "
              f"being joined the way the correlation matrix says, and no power number below it "
              f"means anything.")
        need = next((r["n"] for r in cells if r["power"] >= TARGET_POWER), None)
        print(f"     n for {TARGET_POWER:.0%} power against this difference: "
              + (str(need) if need else f"above {NS[-1]}"))
        print(f"     the scorers agree with each other at rank rho {cells[0]['lat_ab']:+.3f}, "
              f"so the paired test compares correlated rankings, not independent ones")
        print(f"     ceilings for these two runs: {cells[0]['ceilings'][0]:.3f} / "
              f"{cells[0]['ceilings'][1]:.3f}")

    print("\nWhat this does and does not license:")
    print(" - Power is computed with the observed rho as the true rho, the usual plug-in, and")
    print("   it is generous: the true value sits inside an interval this file reproduces, so")
    print("   real power against a weaker truth is lower than printed.")
    print(" - The ceiling is a harder limit than power. A scorer that emits 4 values across 33")
    print("   candidates cannot express a perfect ranking of them, so the loss is in the")
    print("   scorer's resolution, not in the number of candidates or the calibration.")
    print(" - More candidates sharpen agreement with the rubric, not agreement with hiring.")
    print("   The labels are generated from the criteria the pipeline encodes, so n = 250")
    print("   measures the same circularity more precisely.")
    print(" - Rows with n > 33 are bootstrap resamples of the 33 real candidates, scores and")
    print("   labels drawn on the same indices so a candidate stays one candidate. They repeat")
    print("   candidates and inherit their quirks, so they size the design's power rather than")
    print("   predicting a new experiment's result.")
    print(" - A null at n = 33 is not a proof of no effect. It is a statement that this")
    print("   design, on these labels, could not have told the difference apart.")


if __name__ == "__main__":
    main()
