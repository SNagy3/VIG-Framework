from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Callable

import numpy as np

PI = math.pi
PI4 = 4.0 * PI


# ============================================================
# Small utilities
# ============================================================
def cumulative_trapezoid_np(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    dx = np.diff(x)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)
    return out


def bisect_root(func: Callable[[float], float], a: float, b: float, tol: float = 1e-12, maxiter: int = 200) -> float:
    fa = func(a)
    fb = func(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("Root is not bracketed.")
    left, right = a, b
    fleft, fright = fa, fb
    for _ in range(maxiter):
        mid = 0.5 * (left + right)
        fmid = func(mid)
        if abs(fmid) < tol or abs(right - left) < tol:
            return mid
        if fleft * fmid <= 0.0:
            right, fright = mid, fmid
        else:
            left, fleft = mid, fmid
    return 0.5 * (left + right)


def safe_float(x) -> float:
    return float(np.asarray(x).item()) if np.ndim(x) == 0 else float(x)


# ============================================================
# Bardeen metric and VIG branch
# ============================================================
def bardeen_f(r: np.ndarray | float, M: float, g: float) -> np.ndarray | float:
    return 1.0 - (2.0 * M * r**2) / ((r**2 + g**2) ** 1.5)


def horizon_radius_bardeen(M: float, g: float, nscan: int = 12000) -> float:
    if abs(g) < 1e-15:
        return 2.0 * M

    f = lambda r: bardeen_f(r, M, g)
    a = max(1e-12, 0.05 * M)
    b = 4.0 * M
    xs = np.linspace(a, b, nscan)
    vals = np.array([safe_float(f(x)) for x in xs])

    roots = []
    for i in range(len(xs) - 1):
        if vals[i] == 0.0:
            roots.append(float(xs[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            roots.append(bisect_root(lambda r: safe_float(f(r)), float(xs[i]), float(xs[i + 1])))

    if not roots:
        raise RuntimeError(f"No positive Bardeen horizon found for g={g}.")
    return max(roots)


def Q_of_r(r: np.ndarray | float, C: float, fvals: np.ndarray | float) -> np.ndarray | float:
    return C**2 + (PI4 * r**2) ** 2 * fvals


def v_prime_smooth(r: np.ndarray, C: float, fvals: np.ndarray) -> np.ndarray:
    Q = Q_of_r(r, C, fvals)
    if np.any(Q <= 0.0):
        raise ValueError("Q(r) <= 0 on the chosen branch.")
    if np.any(np.abs(fvals) < 1e-14):
        raise ValueError("Grid touches the horizon too closely.")
    return (1.0 - C / np.sqrt(Q)) / fvals


def dV_dr(r: np.ndarray, C: float, fvals: np.ndarray) -> np.ndarray:
    Q = Q_of_r(r, C, fvals)
    if np.any(Q <= 0.0):
        raise ValueError("Q(r) <= 0 on the chosen branch.")
    return (PI4 * r**2) ** 2 / np.sqrt(Q)


def compute_Ccrit_numeric(M: float, g: float, nscan: int = 250000) -> tuple[float, float]:
    r_plus = horizon_radius_bardeen(M, g)
    xs = np.linspace(1e-8, r_plus - 1e-8, nscan)
    fvals = bardeen_f(xs, M, g)
    vals = -((PI4 * xs**2) ** 2 * fvals)
    idx = int(np.argmax(vals))
    r_star = float(xs[idx])
    Ccrit = math.sqrt(max(float(vals[idx]), 0.0))
    return Ccrit, r_star


def find_outer_turning_radius(M: float, g: float, C: float, nscan: int = 12000) -> float:
    r_plus = horizon_radius_bardeen(M, g)

    def q(r: float) -> float:
        fv = safe_float(bardeen_f(r, M, g))
        return C**2 + (PI4 * r**2) ** 2 * fv

    xs = np.linspace(1e-8, r_plus - 1e-8, nscan)
    qvals = np.array([q(x) for x in xs])

    roots = []
    for i in range(len(xs) - 1):
        if qvals[i] == 0.0:
            roots.append(float(xs[i]))
        elif qvals[i] * qvals[i + 1] < 0.0:
            roots.append(bisect_root(q, float(xs[i]), float(xs[i + 1])))

    if not roots:
        raise RuntimeError("No interior turning radius found for this C.")
    return max(roots)


def integrate_slice(
    M: float,
    g: float,
    C: float,
    npts: int = 10000,
    eps_turn: float = 1e-6,
    eps_hor: float = 1e-4,
) -> dict:
    r_plus = horizon_radius_bardeen(M, g)
    r_turn = find_outer_turning_radius(M, g, C)

    if r_turn + eps_turn >= r_plus - eps_hor:
        raise RuntimeError("Turning point too close to horizon for stable integration.")

    r = np.linspace(r_turn + eps_turn, r_plus - eps_hor, npts)
    fvals = bardeen_f(r, M, g)

    vp = v_prime_smooth(r, C, fvals)
    dV = dV_dr(r, C, fvals)

    v = cumulative_trapezoid_np(vp, r)
    V = cumulative_trapezoid_np(dV, r)

    return {
        "M": M,
        "g": g,
        "C": C,
        "r_plus": r_plus,
        "r_turn": r_turn,
        "v_horizon": float(v[-1]),
        "V_total": float(V[-1]),
    }


def build_C_family_asymptotic(Ccrit: float, dmin: float = 1e-6, dmax: float = 1e-1, nC: int = 60) -> np.ndarray:
    deltas = np.logspace(math.log10(dmax), math.log10(dmin), nC)
    return Ccrit * (1.0 - deltas)


def family_slice_scan(
    M: float,
    g: float,
    nC: int = 60,
    dmin: float = 1e-6,
    dmax: float = 1e-1,
    npts: int = 10000,
    eps_turn: float = 1e-6,
    eps_hor: float = 1e-4,
) -> dict:
    Ccrit, r_star = compute_Ccrit_numeric(M, g)
    C_values = build_C_family_asymptotic(Ccrit, dmin=dmin, dmax=dmax, nC=nC)

    rows = []
    for C in C_values:
        delta = 1.0 - (C / Ccrit)
        try:
            out = integrate_slice(M, g, float(C), npts=npts, eps_turn=eps_turn, eps_hor=eps_hor)
            rows.append({
                "C": float(C),
                "delta": float(delta),
                "C_over_Ccrit": float(C / Ccrit),
                "r_turn": out["r_turn"],
                "v_horizon": out["v_horizon"],
                "V_total": out["V_total"],
            })
        except Exception as exc:
            rows.append({
                "C": float(C),
                "delta": float(delta),
                "C_over_Ccrit": float(C / Ccrit),
                "r_turn": math.nan,
                "v_horizon": math.nan,
                "V_total": math.nan,
                "error": str(exc),
            })

    valid = [row for row in rows if not math.isnan(row["v_horizon"]) and not math.isnan(row["V_total"])]
    valid.sort(key=lambda row: row["v_horizon"])

    return {
        "M": M,
        "g": g,
        "Ccrit": Ccrit,
        "r_star": r_star,
        "rows": rows,
        "valid": valid,
    }


def late_time_linear_fit(valid_rows: list[dict], tail_frac: float = 0.20) -> dict:
    if len(valid_rows) < 4:
        raise RuntimeError("Not enough valid rows for a late-time fit.")
    v = np.array([row["v_horizon"] for row in valid_rows], dtype=float)
    V = np.array([row["V_total"] for row in valid_rows], dtype=float)
    start = max(0, int((1.0 - tail_frac) * len(v)))
    vv = v[start:]
    VV = V[start:]
    slope, intercept = np.polyfit(vv, VV, 1)
    return {"slope": float(slope), "intercept": float(intercept), "n_tail": len(vv)}


def secant_slopes(valid_rows: list[dict]) -> np.ndarray:
    if len(valid_rows) < 2:
        return np.array([], dtype=float)
    v = np.array([row["v_horizon"] for row in valid_rows], dtype=float)
    V = np.array([row["V_total"] for row in valid_rows], dtype=float)
    return np.diff(V) / np.diff(v)


def tail_secants(valid_rows: list[dict], k: int = 5) -> np.ndarray:
    slopes = secant_slopes(valid_rows)
    if len(slopes) == 0:
        return np.array([], dtype=float)
    return slopes[-k:]


def summarize_scan(scan: dict, tail_k: int = 5) -> dict:
    M = scan["M"]
    valid = scan["valid"]
    summary = {
        "M": M,
        "g": scan["g"],
        "Ccrit": scan["Ccrit"],
        "r_star": scan["r_star"],
        "n_valid": len(valid),
    }

    if len(valid) >= 4:
        fit = late_time_linear_fit(valid, tail_frac=0.20)
        summary["late_slope"] = fit["slope"]
        summary["late_intercept"] = fit["intercept"]

        slopes = secant_slopes(valid)
        if len(slopes) > 0:
            summary["secant_initial"] = float(slopes[0])
            summary["secant_final"] = float(slopes[-1])

        tsec = tail_secants(valid, k=tail_k)
        if len(tsec) > 0:
            summary["tail_secants"] = [float(x) for x in tsec]
            summary["tail_secant_mean"] = float(np.mean(tsec))
            summary["tail_secant_min"] = float(np.min(tsec))
            summary["tail_secant_max"] = float(np.max(tsec))

    cr_slope = 3.0 * math.sqrt(3.0) * PI * M**2
    summary["cr_slope"] = cr_slope
    if "late_slope" in summary:
        summary["late_ratio_to_cr"] = summary["late_slope"] / cr_slope
        summary["late_dev_to_cr"] = abs(summary["late_slope"] - cr_slope) / cr_slope
    if "tail_secant_mean" in summary:
        summary["tail_mean_ratio_to_cr"] = summary["tail_secant_mean"] / cr_slope
        summary["tail_mean_dev_to_cr"] = abs(summary["tail_secant_mean"] - cr_slope) / cr_slope
        summary["tail_mean_ratio_to_Ccrit"] = summary["tail_secant_mean"] / summary["Ccrit"]
        summary["tail_mean_dev_to_Ccrit"] = abs(summary["tail_secant_mean"] - summary["Ccrit"]) / summary["Ccrit"]

    return summary


# ============================================================
# Exact Bardeen analytic reduction
# ============================================================
def bardeen_f_dimless(x: float, beta: float) -> float:
    return 1.0 - (2.0 * x**2) / ((x**2 + beta**2) ** 1.5)


def horizon_y_polynomial(y: float, beta: float) -> float:
    return y**3 - 2.0 * y**2 + 2.0 * beta**2


def critical_y_polynomial(y: float, beta: float) -> float:
    return 2.0 * y**5 - 3.0 * y**4 + 3.0 * beta**4


def y_to_x(y: float, beta: float) -> float:
    val = y**2 - beta**2
    return math.sqrt(max(val, 0.0))


def bardeen_outer_horizon_dimless(beta: float, nscan: int = 20000) -> tuple[float, float]:
    if abs(beta) < 1e-15:
        return 2.0, 2.0

    ys = np.linspace(max(beta + 1e-10, 1e-10), 4.0, nscan)
    vals = np.array([horizon_y_polynomial(float(y), beta) for y in ys])

    roots = []
    for i in range(len(ys) - 1):
        if vals[i] == 0.0:
            roots.append(float(ys[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            roots.append(bisect_root(lambda y: horizon_y_polynomial(y, beta), float(ys[i]), float(ys[i + 1])))

    if not roots:
        raise RuntimeError(f"No Bardeen horizon root found for beta={beta}.")
    y_plus = max(roots)
    x_plus = y_to_x(y_plus, beta)
    return x_plus, y_plus


def phi_beta_from_y(y: float, beta: float) -> float:
    x = y_to_x(y, beta)
    if x <= 0.0:
        return -float("inf")
    return -x**4 * bardeen_f_dimless(x, beta)


def bardeen_critical_root_dimless(beta: float, nscan: int = 50000) -> tuple[float, float]:
    if abs(beta) < 1e-15:
        return 1.5, 1.5

    x_plus, y_plus = bardeen_outer_horizon_dimless(beta)
    ys = np.linspace(max(beta + 1e-8, 1e-8), y_plus - 1e-8, nscan)
    vals = np.array([critical_y_polynomial(float(y), beta) for y in ys])

    roots = []
    for i in range(len(ys) - 1):
        if vals[i] == 0.0:
            roots.append(float(ys[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            roots.append(bisect_root(lambda y: critical_y_polynomial(y, beta), float(ys[i]), float(ys[i + 1])))

    if not roots:
        raise RuntimeError(f"No critical root found for beta={beta}.")

    best_y = None
    best_phi = -float("inf")
    for y in roots:
        x = y_to_x(y, beta)
        if 0.0 < x < x_plus:
            val = phi_beta_from_y(y, beta)
            if val > best_phi:
                best_phi = val
                best_y = y

    if best_y is None:
        raise RuntimeError(f"No physical maximizing critical root found for beta={beta}.")
    return y_to_x(best_y, beta), best_y


def analytic_Ccrit(M: float, g: float) -> tuple[float, float, float, float, float]:
    beta = g / M
    if abs(beta) < 1e-15:
        ccrit = 3.0 * math.sqrt(3.0) * PI * M * M
        return ccrit, 1.5, 1.5, 2.0, 2.0

    x_plus, y_plus = bardeen_outer_horizon_dimless(beta)
    x_star, y_star = bardeen_critical_root_dimless(beta)
    f_star = bardeen_f_dimless(x_star, beta)
    if f_star >= 0:
        raise RuntimeError(f"Unexpected non-interior critical point for beta={beta}.")
    ccrit = 4.0 * PI * M * M * (x_star**2) * math.sqrt(-f_star)
    return ccrit, x_star, y_star, x_plus, y_plus


def small_beta_Ccrit(M: float, g: float) -> float:
    beta = g / M
    return 3.0 * math.sqrt(3.0) * PI * M * M * (1.0 - (4.0 / 3.0) * beta**2)


def near_extremal_series(M: float, g: float) -> dict:
    beta = g / M
    beta_c = 4.0 / (3.0 * math.sqrt(3.0))
    t = beta_c**2 - beta**2
    if t < 0:
        raise ValueError("Near-extremal expansion requires beta <= beta_c.")

    # y_+ follows the same cubic edge structure as Hayward in the auxiliary variable y
    y_plus = 4.0 / 3.0 + math.sqrt(t) - 0.25 * t + (5.0 / 32.0) * (t ** 1.5)
    # derived from 2 y^5 - 3 y^4 + 3 beta^4 = 0
    y_star = 4.0 / 3.0 + (9.0 / 8.0) * t - (3645.0 / 512.0) * (t ** 2)

    x_plus = math.sqrt(max(y_plus**2 - beta**2, 0.0))
    x_star = math.sqrt(max(y_star**2 - beta**2, 0.0))

    # from series expansion of 4π x_*^2 sqrt(-f(x_*,beta))
    ccrit = ((16.0 * math.sqrt(6.0) * PI) / 9.0) * M * M * math.sqrt(t) \
            + ((21.0 * math.sqrt(6.0) * PI) / 8.0) * M * M * (t ** 1.5) \
            - ((13689.0 * math.sqrt(6.0) * PI) / 2048.0) * M * M * (t ** 2.5)

    return {
        "beta_c": beta_c,
        "t": t,
        "x_plus_ne": x_plus,
        "y_plus_ne": y_plus,
        "x_star_ne": x_star,
        "y_star_ne": y_star,
        "Ccrit_ne": ccrit,
    }


# ============================================================
# Unified Bardeen driver
# ============================================================
def export_rows_csv(rows: list[dict], filepath: str | Path) -> None:
    if not rows:
        return
    path = Path(filepath)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            full_row = {k: row.get(k, math.nan) for k in fieldnames}
            writer.writerow(full_row)


def run_bardeen_unified(
    outdir: str | Path = ".",
    M: float = 10.0,
    g_grid: list[float] | None = None,
    nC: int = 60,
    dmin: float = 1e-6,
    dmax: float = 1e-1,
    npts: int = 12000,
    eps_turn: float = 1e-6,
    eps_hor: float = 1e-4,
    tail_k: int = 5,
    make_plots: bool = True,
) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if g_grid is None:
        g_grid = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]

    rows = []
    for g in g_grid:
        scan = family_slice_scan(
            M=M,
            g=float(g),
            nC=nC,
            dmin=dmin,
            dmax=dmax,
            npts=npts,
            eps_turn=eps_turn,
            eps_hor=eps_hor,
        )
        summary = summarize_scan(scan, tail_k=tail_k)

        ccrit_an, x_star_an, y_star_an, x_plus_an, y_plus_an = analytic_Ccrit(M, float(g))
        ccrit_small = small_beta_Ccrit(M, float(g))
        ne = near_extremal_series(M, float(g))

        rows.append({
            "M": M,
            "g": float(g),
            "beta": float(g / M),
            "beta_c": ne["beta_c"],
            "g_c": ne["beta_c"] * M,
            "g_over_g_c": float(g / (ne["beta_c"] * M)),
            "Ccrit_numeric": summary["Ccrit"],
            "Ccrit_analytic": ccrit_an,
            "Ccrit_small_beta": ccrit_small,
            "Ccrit_near_extremal": ne["Ccrit_ne"],
            "x_star_analytic": x_star_an,
            "y_star_analytic": y_star_an,
            "x_plus_analytic": x_plus_an,
            "y_plus_analytic": y_plus_an,
            "x_star_near_extremal": ne["x_star_ne"],
            "y_star_near_extremal": ne["y_star_ne"],
            "x_plus_near_extremal": ne["x_plus_ne"],
            "y_plus_near_extremal": ne["y_plus_ne"],
            "tail_mean": summary.get("tail_secant_mean", math.nan),
            "tail_mean_ratio_to_cr": summary.get("tail_mean_ratio_to_cr", math.nan),
            "tail_mean_ratio_to_Ccrit": summary.get("tail_mean_ratio_to_Ccrit", math.nan),
            "late_slope": summary.get("late_slope", math.nan),
            "cr_slope": summary["cr_slope"],
            "rel_err_analytic": abs(ccrit_an - summary["Ccrit"]) / summary["Ccrit"],
            "rel_err_small_beta": abs(ccrit_small - summary["Ccrit"]) / summary["Ccrit"],
            "rel_err_near_extremal": abs(ne["Ccrit_ne"] - summary["Ccrit"]) / summary["Ccrit"],
            "t_near_extremal": ne["t"],
        })

    master_csv = outdir / "vig_bardeen_master_summary.csv"
    atlas_csv = outdir / "vig_bardeen_atlas.csv"
    export_rows_csv(rows, master_csv)
    export_rows_csv(
        [{
            "M": row["M"],
            "g": row["g"],
            "beta": row["beta"],
            "x_star_analytic": row["x_star_analytic"],
            "x_plus_analytic": row["x_plus_analytic"],
            "Ccrit_analytic": row["Ccrit_analytic"],
            "Ccrit_numeric": row["Ccrit_numeric"],
            "tail_mean": row["tail_mean"],
        } for row in rows],
        atlas_csv
    )

    plot_files = []
    if make_plots:
        try:
            import matplotlib.pyplot as plt

            beta = np.array([r["beta"] for r in rows], dtype=float)
            x_star = np.array([r["x_star_analytic"] for r in rows], dtype=float)
            x_plus = np.array([r["x_plus_analytic"] for r in rows], dtype=float)
            c_beta = np.array([r["Ccrit_analytic"] / (4.0 * PI * (M**2)) for r in rows], dtype=float)
            ccrit_over_cr = np.array([r["Ccrit_analytic"] / (3.0 * math.sqrt(3.0) * PI * (M**2)) for r in rows], dtype=float)
            c_an = np.array([r["Ccrit_analytic"] for r in rows], dtype=float)
            c_sm = np.array([r["Ccrit_small_beta"] for r in rows], dtype=float)
            c_ne = np.array([r["Ccrit_near_extremal"] for r in rows], dtype=float)
            tail = np.array([r["tail_mean"] for r in rows], dtype=float)

            plt.figure(figsize=(8, 5))
            plt.plot(beta, x_star, label=r"$x_*(\beta)$")
            plt.plot(beta, x_plus, label=r"$x_+(\beta)$")
            plt.xlabel(r"$\beta=g/M$")
            plt.ylabel("dimensionless radius")
            plt.title(r"Bardeen interior atlas: $x_*(\beta)$ and $x_+(\beta)$")
            plt.legend()
            plt.tight_layout()
            p = outdir / "vig_bardeen_atlas_xstar_xplus.png"
            plt.savefig(p, dpi=200)
            plt.close()
            plot_files.append(str(p))

            plt.figure(figsize=(8, 5))
            plt.plot(beta, c_beta)
            plt.xlabel(r"$\beta=g/M$")
            plt.ylabel(r"$\mathcal{B}(\beta)$")
            plt.title(r"Bardeen interior atlas: $\mathcal{B}(\beta)$")
            plt.tight_layout()
            p = outdir / "vig_bardeen_atlas_B_beta.png"
            plt.savefig(p, dpi=200)
            plt.close()
            plot_files.append(str(p))

            plt.figure(figsize=(8, 5))
            plt.plot(beta, ccrit_over_cr)
            plt.xlabel(r"$\beta=g/M$")
            plt.ylabel(r"$C_{\rm crit}/(3\sqrt{3}\pi M^2)$")
            plt.title(r"Bardeen interior atlas: normalized critical growth law")
            plt.tight_layout()
            p = outdir / "vig_bardeen_atlas_Ccrit_over_CR.png"
            plt.savefig(p, dpi=200)
            plt.close()
            plot_files.append(str(p))

            plt.figure(figsize=(8, 5))
            plt.plot(beta, c_an, label="analytic")
            plt.plot(beta, c_sm, label="small-$\\beta$")
            plt.plot(beta, c_ne, label="near-extremal")
            plt.xlabel(r"$\beta=g/M$")
            plt.ylabel(r"$C_{\rm crit}(M,g)$")
            plt.title(r"Bardeen: exact law vs asymptotic formulas")
            plt.legend()
            plt.tight_layout()
            p = outdir / "vig_bardeen_exact_vs_asymptotics.png"
            plt.savefig(p, dpi=200)
            plt.close()
            plot_files.append(str(p))

            plt.figure(figsize=(8, 5))
            plt.plot(beta, c_an, label=r"$C_{\rm crit}$")
            if np.any(~np.isnan(tail)):
                plt.plot(beta, tail, label="tail mean")
            plt.xlabel(r"$\beta=g/M$")
            plt.ylabel(r"$C_{\rm crit}$ / tail mean")
            plt.title(r"Bardeen: critical growth law vs tail mean")
            plt.legend()
            plt.tight_layout()
            p = outdir / "vig_bardeen_ccrit_vs_tail.png"
            plt.savefig(p, dpi=200)
            plt.close()
            plot_files.append(str(p))
        except Exception:
            pass

    return {
        "master_csv": str(master_csv),
        "atlas_csv": str(atlas_csv),
        "plots": plot_files,
        "n_rows": len(rows),
    }


if __name__ == "__main__":
    result = run_bardeen_unified(outdir=".", M=10.0, make_plots=True)
    print("saved:")
    print(f"  {result['master_csv']}")
    print(f"  {result['atlas_csv']}")
    for p in result["plots"]:
        print(f"  {p}")
    print(f"rows = {result['n_rows']}")
