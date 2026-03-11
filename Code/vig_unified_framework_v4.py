from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Callable

import numpy as np

PI = math.pi
PI4 = 4.0 * PI


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


# -----------------------------
# Clock / ansatz layer
# -----------------------------
M_SUN_PL = 9.135e37

def universal_clock_row(M_Msun: float, C: float = 4.0 * PI) -> dict:
    if M_Msun <= 0:
        raise ValueError("M_Msun must be positive.")
    if C <= 0:
        raise ValueError("C must be positive.")
    M_pl = M_Msun * M_SUN_PL
    S = 4.0 * PI * M_pl**2
    T_pl = 1.0 / (8.0 * PI * M_pl)
    log10_tau_sat_pl = math.log10(C * M_pl) + S * math.log10(math.e)
    invariant_I = C / (8.0 * PI)
    return {
        "Mass_Msun": M_Msun,
        "Mass_Pl": M_pl,
        "Entropy_S": S,
        "Temperature_T_pl": T_pl,
        "Coeff_C": C,
        "Invariant_I": invariant_I,
        "Log10_TauSat_Pl": log10_tau_sat_pl,
    }


# -----------------------------
# Metrics
# -----------------------------
def schwarzschild_f(r: np.ndarray | float, M: float) -> np.ndarray | float:
    return 1.0 - 2.0 * M / r


def hayward_f(r: np.ndarray | float, M: float, ell: float) -> np.ndarray | float:
    return 1.0 - (2.0 * M * r**2) / (r**3 + 2.0 * M * ell**2)


def get_metric_f(metric: str):
    metric = metric.lower()
    if metric == "schwarzschild":
        return lambda r, M, ell: schwarzschild_f(r, M)
    if metric == "hayward":
        return hayward_f
    raise ValueError(f"Unknown metric: {metric}")


def horizon_radius(metric: str, M: float, ell: float) -> float:
    metric = metric.lower()
    if metric == "schwarzschild":
        return 2.0 * M
    if metric == "hayward":
        f = lambda r: hayward_f(r, M, ell)
        a = max(1e-12, 0.05 * M)
        b = 4.0 * M
        xs = np.linspace(a, b, 8000)
        vals = np.array([f(x) for x in xs])
        roots = []
        for i in range(len(xs) - 1):
            if vals[i] == 0.0:
                roots.append(xs[i])
            elif vals[i] * vals[i + 1] < 0.0:
                roots.append(bisect_root(f, xs[i], xs[i + 1]))
        if not roots:
            raise RuntimeError("No positive Hayward horizon found.")
        return max(roots)
    raise ValueError(f"Unknown metric: {metric}")


# -----------------------------
# VIG branch
# -----------------------------
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


# -----------------------------
# Critical constant and scans
# -----------------------------
def compute_Ccrit_numeric(metric: str, M: float, ell: float, nscan: int = 250000) -> tuple[float, float]:
    f_metric = get_metric_f(metric)
    r_plus = horizon_radius(metric, M, ell)
    xs = np.linspace(1e-8, r_plus - 1e-8, nscan)
    fvals = f_metric(xs, M, ell)
    vals = -((PI4 * xs**2) ** 2 * fvals)
    idx = int(np.argmax(vals))
    return math.sqrt(max(float(vals[idx]), 0.0)), float(xs[idx])


def find_outer_turning_radius(metric: str, M: float, ell: float, C: float, nscan: int = 12000) -> float:
    f_metric = get_metric_f(metric)
    r_plus = horizon_radius(metric, M, ell)

    def q(r: float) -> float:
        fv = float(f_metric(r, M, ell))
        return C**2 + (PI4 * r**2) ** 2 * fv

    xs = np.linspace(1e-8, r_plus - 1e-8, nscan)
    qvals = np.array([q(x) for x in xs])
    roots = []
    for i in range(len(xs) - 1):
        if qvals[i] == 0.0:
            roots.append(xs[i])
        elif qvals[i] * qvals[i + 1] < 0.0:
            roots.append(bisect_root(q, xs[i], xs[i + 1]))
    if not roots:
        raise RuntimeError("No interior turning radius found for this C.")
    return max(roots)


def integrate_slice(metric: str, M: float, ell: float, C: float,
                    npts: int = 10000, eps_turn: float = 1e-6, eps_hor: float = 1e-4) -> dict:
    f_metric = get_metric_f(metric)
    r_plus = horizon_radius(metric, M, ell)
    r_turn = find_outer_turning_radius(metric, M, ell, C)
    if r_turn + eps_turn >= r_plus - eps_hor:
        raise RuntimeError("Turning point too close to horizon for stable integration.")
    r = np.linspace(r_turn + eps_turn, r_plus - eps_hor, npts)
    fvals = f_metric(r, M, ell)
    vp = v_prime_smooth(r, C, fvals)
    dV = dV_dr(r, C, fvals)
    v = cumulative_trapezoid_np(vp, r)
    V = cumulative_trapezoid_np(dV, r)
    return {
        "metric": metric, "M": M, "ell": ell, "C": C,
        "r_plus": r_plus, "r_turn": r_turn,
        "v_horizon": float(v[-1]), "V_total": float(V[-1]),
    }


def build_C_family_asymptotic(Ccrit: float, dmin: float = 1e-6, dmax: float = 1e-1, nC: int = 60) -> np.ndarray:
    deltas = np.logspace(math.log10(dmax), math.log10(dmin), nC)
    return Ccrit * (1.0 - deltas)


def family_slice_scan(metric: str, M: float, ell: float = 0.0, nC: int = 60,
                      dmin: float = 1e-6, dmax: float = 1e-1, npts: int = 10000,
                      eps_turn: float = 1e-6, eps_hor: float = 1e-4) -> dict:
    Ccrit, r_star = compute_Ccrit_numeric(metric, M, ell)
    C_values = build_C_family_asymptotic(Ccrit, dmin=dmin, dmax=dmax, nC=nC)
    rows = []
    for C in C_values:
        delta = 1.0 - (C / Ccrit)
        try:
            out = integrate_slice(metric, M, ell, float(C), npts=npts, eps_turn=eps_turn, eps_hor=eps_hor)
            rows.append({
                "C": float(C), "delta": float(delta), "C_over_Ccrit": float(C / Ccrit),
                "r_turn": out["r_turn"], "v_horizon": out["v_horizon"], "V_total": out["V_total"],
            })
        except Exception as exc:
            rows.append({
                "C": float(C), "delta": float(delta), "C_over_Ccrit": float(C / Ccrit),
                "r_turn": math.nan, "v_horizon": math.nan, "V_total": math.nan, "error": str(exc),
            })
    valid = [row for row in rows if not math.isnan(row["v_horizon"]) and not math.isnan(row["V_total"])]
    valid.sort(key=lambda row: row["v_horizon"])
    return {"metric": metric, "M": M, "ell": ell, "Ccrit": Ccrit, "r_star": r_star, "rows": rows, "valid": valid}


def late_time_linear_fit(valid_rows: list[dict], tail_frac: float = 0.20) -> dict:
    if len(valid_rows) < 4:
        raise RuntimeError("Not enough valid rows for a late-time fit.")
    v = np.array([row["v_horizon"] for row in valid_rows], dtype=float)
    V = np.array([row["V_total"] for row in valid_rows], dtype=float)
    start = max(0, int((1.0 - tail_frac) * len(v)))
    vv, VV = v[start:], V[start:]
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
    return np.array([], dtype=float) if len(slopes) == 0 else slopes[-k:]


def summarize_scan(scan: dict, tail_k: int = 5) -> dict:
    M = scan["M"]
    valid = scan["valid"]
    summary = {
        "metric": scan["metric"], "M": M, "ell": scan["ell"],
        "Ccrit": scan["Ccrit"], "r_star": scan["r_star"], "n_valid": len(valid),
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
    cr_slope = 3.0 * math.sqrt(3.0) * PI * M**2
    summary["cr_slope"] = cr_slope
    if "late_slope" in summary:
        summary["ratio_to_cr"] = summary["late_slope"] / cr_slope
    if "tail_secant_mean" in summary:
        summary["tail_mean_ratio_to_cr"] = summary["tail_secant_mean"] / cr_slope
        summary["tail_mean_ratio_to_Ccrit"] = summary["tail_secant_mean"] / summary["Ccrit"]
    return summary


# -----------------------------
# Analytic Ccrit
# -----------------------------
def hayward_f_dimless(x: float, lam: float) -> float:
    return 1.0 - (2.0 * x * x) / (x**3 + 2.0 * lam * lam)


def hayward_horizon_polynomial(x: float, lam: float) -> float:
    return x**3 - 2.0 * x**2 + 2.0 * lam * lam


def hayward_outer_horizon_dimless(lam: float, nscan: int = 20000) -> float:
    xs = np.linspace(1e-10, 4.0, nscan)
    vals = xs**3 - 2.0 * xs**2 + 2.0 * lam * lam
    roots = []
    for i in range(len(xs) - 1):
        if vals[i] == 0.0:
            roots.append(float(xs[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            f = lambda x: hayward_horizon_polynomial(x, lam)
            roots.append(bisect_root(f, float(xs[i]), float(xs[i + 1])))
    if not roots:
        raise RuntimeError(f"No outer horizon found for lambda={lam}")
    return max(roots)


def phi_lambda(x: float, lam: float) -> float:
    return -x**4 * hayward_f_dimless(x, lam)


def critical_polynomial(x: float, lam: float) -> float:
    return 2.0*x**6 - 3.0*x**5 + 8.0*(lam**2)*x**3 - 12.0*(lam**2)*x**2 + 8.0*(lam**4)


def critical_root_dimless(lam: float, nscan: int = 50000) -> float:
    x_plus = hayward_outer_horizon_dimless(lam)
    xs = np.linspace(1e-8, x_plus - 1e-8, nscan)
    vals = np.array([critical_polynomial(float(x), lam) for x in xs])
    roots = []
    for i in range(len(xs) - 1):
        if vals[i] == 0.0:
            roots.append(float(xs[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            f = lambda x: critical_polynomial(x, lam)
            roots.append(bisect_root(f, float(xs[i]), float(xs[i + 1])))
    if not roots:
        raise RuntimeError(f"No critical root found for lambda={lam}")
    best_root, best_phi = None, -float("inf")
    for root in roots:
        if 0.0 < root < x_plus:
            val = phi_lambda(root, lam)
            if val > best_phi:
                best_phi = val
                best_root = root
    if best_root is None:
        raise RuntimeError(f"No physical maximizing root found for lambda={lam}")
    return best_root


def analytic_Ccrit(M: float, ell: float) -> tuple[float, float, float]:
    lam = ell / M
    if abs(lam) < 1e-15:
        return 3.0 * math.sqrt(3.0) * PI * M * M, 1.5, 2.0
    x_plus = hayward_outer_horizon_dimless(lam)
    x_star = critical_root_dimless(lam)
    f_star = hayward_f_dimless(x_star, lam)
    ccrit = 4.0 * PI * M * M * (x_star**2) * math.sqrt(-f_star)
    return ccrit, x_star, x_plus


def small_lambda_Ccrit(M: float, ell: float) -> float:
    lam = ell / M
    return 3.0 * math.sqrt(3.0) * PI * M * M * (1.0 - (32.0/27.0) * lam * lam)


def near_extremal_series(M: float, ell: float) -> dict:
    lam = ell / M
    lam_c = 4.0 / (3.0 * math.sqrt(3.0))
    t = lam_c**2 - lam**2
    if t < 0:
        raise ValueError("Requires lambda <= lambda_c.")
    x_plus = 4.0/3.0 + math.sqrt(t) - 0.25*t + (5.0/32.0)*(t**1.5)
    x_star = 4.0/3.0 + 0.75*t - (135.0/64.0)*(t**2)
    ccrit = (16.0*PI/3.0)*M*M*math.sqrt(t) + 3.0*PI*M*M*(t**1.5) - (99.0*PI/32.0)*M*M*(t**2.5)
    ccrit_norm = (16.0/(9.0*math.sqrt(3.0)))*math.sqrt(t) + (1.0/math.sqrt(3.0))*(t**1.5)
    return {"lambda_c": lam_c, "t": t, "x_plus_ne": x_plus, "x_star_ne": x_star, "Ccrit_ne": ccrit, "Ccrit_norm_ne": ccrit_norm}


# -----------------------------
# Unified driver
# -----------------------------
def build_ell_grid(M: float, step: float = 0.25) -> tuple[np.ndarray, float]:
    ell_crit = 4.0 * M / (3.0 * math.sqrt(3.0))
    ell_max = 0.99 * ell_crit
    n = int(math.floor(ell_max / step)) + 1
    ells = np.round(np.linspace(0.0, n * step, n + 1), 10)
    ells = ells[ells <= ell_max + 1e-12]
    if len(ells) == 0 or ells[0] != 0.0:
        ells = np.insert(ells, 0, 0.0)
    return ells, ell_crit


def run_full_framework(outdir: str | Path = ".", M: float = 10.0, ell_step: float = 0.25,
                       nC: int = 60, dmin: float = 1e-6, dmax: float = 1e-1,
                       npts: int = 12000, eps_turn: float = 1e-6, eps_hor: float = 1e-4,
                       tail_k: int = 5, make_plots: bool = True) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ells, ell_crit = build_ell_grid(M, step=ell_step)
    summary_rows = []

    for ell in ells:
        metric = "schwarzschild" if abs(ell) < 1e-15 else "hayward"
        scan = family_slice_scan(metric=metric, M=M, ell=float(ell), nC=nC, dmin=dmin, dmax=dmax,
                                 npts=npts, eps_turn=eps_turn, eps_hor=eps_hor)
        summary = summarize_scan(scan, tail_k=tail_k)
        ccrit_an, x_star_an, x_plus_an = analytic_Ccrit(M, float(ell))
        ccrit_small = small_lambda_Ccrit(M, float(ell))
        ne = near_extremal_series(M, float(ell))
        summary_rows.append({
            "metric": metric, "M": M, "ell": float(ell), "lambda": float(ell / M),
            "ell_crit": ell_crit, "ell_over_ellcrit": float(ell / ell_crit),
            "Ccrit_numeric": summary["Ccrit"], "Ccrit_analytic": ccrit_an,
            "Ccrit_small_lambda": ccrit_small, "Ccrit_near_extremal": ne["Ccrit_ne"],
            "x_star_analytic": x_star_an, "x_plus_analytic": x_plus_an,
            "x_star_near_extremal": ne["x_star_ne"], "x_plus_near_extremal": ne["x_plus_ne"],
            "tail_mean": summary.get("tail_secant_mean", math.nan),
            "tail_mean_ratio_to_cr": summary.get("tail_mean_ratio_to_cr", math.nan),
            "tail_mean_ratio_to_Ccrit": summary.get("tail_mean_ratio_to_Ccrit", math.nan),
            "late_slope": summary.get("late_slope", math.nan), "cr_slope": summary["cr_slope"],
            "rel_err_analytic": abs(ccrit_an - summary["Ccrit"]) / summary["Ccrit"],
            "rel_err_small_lambda": abs(ccrit_small - summary["Ccrit"]) / summary["Ccrit"],
            "rel_err_near_extremal": abs(ne["Ccrit_ne"] - summary["Ccrit"]) / summary["Ccrit"],
            "t_near_extremal": ne["t"],
        })

    master_csv = outdir / "vig_unified_master_summary.csv"
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    atlas_csv = outdir / "vig_interior_atlas_unified.csv"
    with open(atlas_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric", "M", "ell", "lambda", "x_star_analytic", "x_plus_analytic",
                      "Ccrit_analytic", "Ccrit_numeric", "tail_mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row[k] for k in fieldnames})

    plot_files = []
    if make_plots:
        try:
            import matplotlib.pyplot as plt
            rows = summary_rows
            lam = np.array([r["lambda"] for r in rows], dtype=float)
            x_star = np.array([r["x_star_analytic"] for r in rows], dtype=float)
            x_plus = np.array([r["x_plus_analytic"] for r in rows], dtype=float)
            c_lambda = np.array([r["Ccrit_analytic"] / (4.0 * PI * (M**2)) for r in rows], dtype=float)
            ccrit_over_cr = np.array([r["Ccrit_analytic"] / (3.0 * math.sqrt(3.0) * PI * (M**2)) for r in rows], dtype=float)
            c_an = np.array([r["Ccrit_analytic"] for r in rows], dtype=float)
            c_sm = np.array([r["Ccrit_small_lambda"] for r in rows], dtype=float)
            c_ne = np.array([r["Ccrit_near_extremal"] for r in rows], dtype=float)

            plt.figure(figsize=(8, 5))
            plt.plot(lam, x_star, label=r"$x_*(\lambda)$")
            plt.plot(lam, x_plus, label=r"$x_+(\lambda)$")
            plt.xlabel(r"$\lambda=\ell/M$")
            plt.ylabel("dimensionless radius")
            plt.title(r"VIG interior atlas: $x_*(\lambda)$ and $x_+(\lambda)$")
            plt.legend()
            plt.tight_layout()
            p = outdir / "vig_unified_atlas_xstar_xplus.png"
            plt.savefig(p, dpi=200); plt.close(); plot_files.append(str(p))

            plt.figure(figsize=(8, 5))
            plt.plot(lam, c_lambda)
            plt.xlabel(r"$\lambda=\ell/M$")
            plt.ylabel(r"$\mathcal{C}(\lambda)$")
            plt.title(r"VIG interior atlas: $\mathcal{C}(\lambda)$")
            plt.tight_layout()
            p = outdir / "vig_unified_atlas_C_lambda.png"
            plt.savefig(p, dpi=200); plt.close(); plot_files.append(str(p))

            plt.figure(figsize=(8, 5))
            plt.plot(lam, ccrit_over_cr)
            plt.xlabel(r"$\lambda=\ell/M$")
            plt.ylabel(r"$C_{\rm crit}/(3\sqrt{3}\pi M^2)$")
            plt.title(r"VIG interior atlas: normalized critical growth law")
            plt.tight_layout()
            p = outdir / "vig_unified_atlas_Ccrit_over_CR.png"
            plt.savefig(p, dpi=200); plt.close(); plot_files.append(str(p))

            plt.figure(figsize=(8, 5))
            plt.plot(lam, c_an, label="analytic")
            plt.plot(lam, c_sm, label="small-$\\lambda$")
            plt.plot(lam, c_ne, label="near-extremal")
            plt.xlabel(r"$\lambda=\ell/M$")
            plt.ylabel(r"$C_{\rm crit}(M,\ell)$")
            plt.title(r"VIG: exact law vs asymptotic formulas")
            plt.legend()
            plt.tight_layout()
            p = outdir / "vig_unified_exact_vs_asymptotics.png"
            plt.savefig(p, dpi=200); plt.close(); plot_files.append(str(p))
        except Exception:
            pass

    return {"outdir": str(outdir), "master_csv": str(master_csv), "atlas_csv": str(atlas_csv),
            "plots": plot_files, "ell_crit": ell_crit, "n_rows": len(summary_rows)}


if __name__ == "__main__":
    result = run_full_framework(outdir=".", M=10.0, make_plots=True)
    print("saved:")
    print(f"  {result['master_csv']}")
    print(f"  {result['atlas_csv']}")
    for p in result["plots"]:
        print(f"  {p}")
    print(f"ell_crit = {result['ell_crit']:.12f}")
    print(f"rows     = {result['n_rows']}")
