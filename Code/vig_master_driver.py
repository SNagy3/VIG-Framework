from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PI = math.pi
M_SUN_PL = 9.135e37


# ============================================================
# Small utilities
# ============================================================
def export_rows_csv(rows: list[dict], filepath: str | Path) -> None:
    if not rows:
        return
    path = Path(filepath)
    fieldnames: list[str] = []
    seen: set[str] = set()
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


def _require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


# ============================================================
# Universal Clock block
# ============================================================
def universal_clock(M_Msun: float, C: float = 4.0 * PI) -> dict:
    """
    Conjectural interior clock bookkeeping in Planck units.

    Ansatz:
        tau_sat = C * M * exp(S)
        S = 4*pi*M^2
        T = 1/(8*pi*M)

    Hence:
        I = tau_sat * T / exp(S) = C / (8*pi)
    """
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


def build_universal_clock_summary() -> pd.DataFrame:
    masses = [6.3, 21.2, 4.3e6, 6.5e9, 6.6e10]
    rows = [universal_clock(m, C=4.0 * PI) for m in masses]
    return pd.DataFrame(rows)


# ============================================================
# Hayward loaders
# ============================================================
def load_hayward_full(base: Path) -> pd.DataFrame:
    """
    Preferred sources, in order:
      1. vig_unified_master_summary.csv
      2. vig_interior_atlas_unified.csv
      3. vig_interior_atlas.csv
      4. vig_analytic_ccrit_validation.csv
    """
    candidates = [
        base / "vig_unified_master_summary.csv",
        base / "vig_interior_atlas_unified.csv",
        base / "vig_interior_atlas.csv",
        base / "vig_analytic_ccrit_validation.csv",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            "Could not find a Hayward source. Expected one of:\n"
            "  vig_unified_master_summary.csv\n"
            "  vig_interior_atlas_unified.csv\n"
            "  vig_interior_atlas.csv\n"
            "  vig_analytic_ccrit_validation.csv"
        )

    df = pd.read_csv(src)

    if src.name == "vig_unified_master_summary.csv":
        _require_columns(
            df,
            ["metric", "M", "ell", "lambda", "x_star_analytic", "x_plus_analytic", "Ccrit_analytic"],
            src.name,
        )
        # Keep only Hayward rows if metric exists
        if "metric" in df.columns:
            df = df[df["metric"].astype(str).str.lower().eq("hayward")].copy()

        out = pd.DataFrame({
            "model": "Hayward",
            "M": df["M"].astype(float),
            "param": df["ell"].astype(float),
            "u": df["lambda"].astype(float),
            "x_star": df["x_star_analytic"].astype(float),
            "x_plus": df["x_plus_analytic"].astype(float),
            "Ccrit_exact": df["Ccrit_analytic"].astype(float),
            "Ccrit_numeric": df.get("Ccrit_numeric", df["Ccrit_analytic"]).astype(float),
            "tail_mean": df.get("tail_mean", np.nan),
        })
        return out.drop_duplicates(subset=["u"]).sort_values("u").reset_index(drop=True)

    if src.name == "vig_interior_atlas_unified.csv":
        _require_columns(
            df,
            ["M", "ell", "lambda", "x_star_analytic", "x_plus_analytic", "Ccrit_analytic"],
            src.name,
        )
        out = pd.DataFrame({
            "model": "Hayward",
            "M": df["M"].astype(float),
            "param": df["ell"].astype(float),
            "u": df["lambda"].astype(float),
            "x_star": df["x_star_analytic"].astype(float),
            "x_plus": df["x_plus_analytic"].astype(float),
            "Ccrit_exact": df["Ccrit_analytic"].astype(float),
            "Ccrit_numeric": df.get("Ccrit_numeric", df["Ccrit_analytic"]).astype(float),
            "tail_mean": df.get("tail_mean", np.nan),
        })
        return out.drop_duplicates(subset=["u"]).sort_values("u").reset_index(drop=True)

    if src.name == "vig_interior_atlas.csv":
        _require_columns(
            df,
            ["M", "ell", "lambda", "x_star", "x_plus", "Ccrit_analytic"],
            src.name,
        )
        out = pd.DataFrame({
            "model": "Hayward",
            "M": df["M"].astype(float),
            "param": df["ell"].astype(float),
            "u": df["lambda"].astype(float),
            "x_star": df["x_star"].astype(float),
            "x_plus": df["x_plus"].astype(float),
            "Ccrit_exact": df["Ccrit_analytic"].astype(float),
            "Ccrit_numeric": df.get("Ccrit_numeric", df["Ccrit_analytic"]).astype(float),
            "tail_mean": df.get("tail_mean", np.nan),
        })
        return out.drop_duplicates(subset=["u"]).sort_values("u").reset_index(drop=True)

    _require_columns(
        df,
        ["M", "ell", "lambda", "x_star_analytic", "x_plus_analytic", "Ccrit_analytic", "Ccrit_numeric"],
        src.name,
    )
    out = pd.DataFrame({
        "model": "Hayward",
        "M": df["M"].astype(float),
        "param": df["ell"].astype(float),
        "u": df["lambda"].astype(float),
        "x_star": df["x_star_analytic"].astype(float),
        "x_plus": df["x_plus_analytic"].astype(float),
        "Ccrit_exact": df["Ccrit_analytic"].astype(float),
        "Ccrit_numeric": df["Ccrit_numeric"].astype(float),
        "tail_mean": np.nan,
    })
    return out.drop_duplicates(subset=["u"]).sort_values("u").reset_index(drop=True)


# ============================================================
# Bardeen loaders
# ============================================================
def load_bardeen_full(base: Path) -> pd.DataFrame:
    """
    Preferred sources, in order:
      1. vig_bardeen_edge_master_summary.csv
      2. vig_bardeen_edge_atlas.csv
      3. vig_bardeen_master_summary.csv
      4. vig_bardeen_atlas.csv
    """
    candidates = [
        base / "vig_bardeen_edge_master_summary.csv",
        base / "vig_bardeen_edge_atlas.csv",
        base / "vig_bardeen_master_summary.csv",
        base / "vig_bardeen_atlas.csv",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            "Could not find a Bardeen source. Expected one of:\n"
            "  vig_bardeen_edge_master_summary.csv\n"
            "  vig_bardeen_edge_atlas.csv\n"
            "  vig_bardeen_master_summary.csv\n"
            "  vig_bardeen_atlas.csv"
        )

    df = pd.read_csv(src)
    if "beta" not in df.columns:
        raise ValueError(f"{src.name} does not contain a 'beta' column.")

    _require_columns(df, ["M", "g", "beta", "x_star_analytic", "x_plus_analytic", "Ccrit_analytic"], src.name)

    out = pd.DataFrame({
        "model": "Bardeen",
        "M": df["M"].astype(float),
        "param": df["g"].astype(float),
        "u": df["beta"].astype(float),
        "x_star": df["x_star_analytic"].astype(float),
        "x_plus": df["x_plus_analytic"].astype(float),
        "Ccrit_exact": df["Ccrit_analytic"].astype(float),
        "Ccrit_numeric": df.get("Ccrit_numeric", df["Ccrit_analytic"]).astype(float),
        "tail_mean": df.get("tail_mean", np.nan),
    })
    return out.drop_duplicates(subset=["u"]).sort_values("u").reset_index(drop=True)


# ============================================================
# Derived columns and comparison grid
# ============================================================
def augment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CR_scale"] = 3.0 * math.sqrt(3.0) * PI * (out["M"] ** 2)
    out["shape_fn"] = out["Ccrit_exact"] / (4.0 * PI * (out["M"] ** 2))
    out["Ccrit_over_CR"] = out["Ccrit_exact"] / out["CR_scale"]
    out["tail_over_CR"] = out["tail_mean"] / out["CR_scale"]
    out["tail_over_Ccrit"] = out["tail_mean"] / out["Ccrit_exact"]
    out["gap"] = out["x_plus"] - out["x_star"]
    return out


def build_common_grid(hay: pd.DataFrame, bar: pd.DataFrame, n: int = 400) -> pd.DataFrame:
    u_min = max(hay["u"].min(), bar["u"].min())
    u_max = min(hay["u"].max(), bar["u"].max())
    if u_max <= u_min:
        raise ValueError("No overlapping normalized-parameter range between Hayward and Bardeen.")

    u = np.linspace(u_min, u_max, n)

    def interp(df: pd.DataFrame, col: str) -> np.ndarray:
        return np.interp(u, df["u"].to_numpy(), df[col].to_numpy())

    out = pd.DataFrame({
        "u": u,
        "hayward_x_star": interp(hay, "x_star"),
        "bardeen_x_star": interp(bar, "x_star"),
        "hayward_x_plus": interp(hay, "x_plus"),
        "bardeen_x_plus": interp(bar, "x_plus"),
        "hayward_gap": interp(hay, "gap"),
        "bardeen_gap": interp(bar, "gap"),
        "hayward_shape_fn": interp(hay, "shape_fn"),
        "bardeen_shape_fn": interp(bar, "shape_fn"),
        "hayward_Ccrit_over_CR": interp(hay, "Ccrit_over_CR"),
        "bardeen_Ccrit_over_CR": interp(bar, "Ccrit_over_CR"),
    })
    out["bardeen_over_hayward_shape_fn"] = out["bardeen_shape_fn"] / out["hayward_shape_fn"]
    out["bardeen_over_hayward_Ccrit_over_CR"] = out["bardeen_Ccrit_over_CR"] / out["hayward_Ccrit_over_CR"]
    out["delta_x_star"] = out["bardeen_x_star"] - out["hayward_x_star"]
    out["delta_x_plus"] = out["bardeen_x_plus"] - out["hayward_x_plus"]
    out["delta_gap"] = out["bardeen_gap"] - out["hayward_gap"]
    return out


# ============================================================
# Plotting
# ============================================================
def make_plots(outdir: Path, hay: pd.DataFrame, bar: pd.DataFrame, common: pd.DataFrame) -> list[str]:
    files: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return files

    plt.figure(figsize=(8, 5))
    plt.plot(hay["u"], hay["x_star"], label="Hayward")
    plt.plot(bar["u"], bar["x_star"], label="Bardeen")
    plt.xlabel(r"normalized deformation $u$")
    plt.ylabel(r"$x_*$")
    plt.title(r"Hayward vs Bardeen (full edge): critical slice radius")
    plt.legend()
    plt.tight_layout()
    p = outdir / "master_compare_edge_xstar_overlay.png"
    plt.savefig(p, dpi=200)
    plt.close()
    files.append(str(p))

    plt.figure(figsize=(8, 5))
    plt.plot(hay["u"], hay["x_plus"], label="Hayward")
    plt.plot(bar["u"], bar["x_plus"], label="Bardeen")
    plt.xlabel(r"normalized deformation $u$")
    plt.ylabel(r"$x_+$")
    plt.title(r"Hayward vs Bardeen (full edge): outer horizon radius")
    plt.legend()
    plt.tight_layout()
    p = outdir / "master_compare_edge_xplus_overlay.png"
    plt.savefig(p, dpi=200)
    plt.close()
    files.append(str(p))

    plt.figure(figsize=(8, 5))
    plt.plot(hay["u"], hay["gap"], label="Hayward")
    plt.plot(bar["u"], bar["gap"], label="Bardeen")
    plt.xlabel(r"normalized deformation $u$")
    plt.ylabel(r"$x_+ - x_*$")
    plt.title(r"Hayward vs Bardeen (full edge): interior channel gap")
    plt.legend()
    plt.tight_layout()
    p = outdir / "master_compare_edge_gap_overlay.png"
    plt.savefig(p, dpi=200)
    plt.close()
    files.append(str(p))

    plt.figure(figsize=(8, 5))
    plt.plot(hay["u"], hay["shape_fn"], label="Hayward")
    plt.plot(bar["u"], bar["shape_fn"], label="Bardeen")
    plt.xlabel(r"normalized deformation $u$")
    plt.ylabel(r"$C_{\rm crit}/(4\pi M^2)$")
    plt.title(r"Hayward vs Bardeen (full edge): shape function")
    plt.legend()
    plt.tight_layout()
    p = outdir / "master_compare_edge_shape_function_overlay.png"
    plt.savefig(p, dpi=200)
    plt.close()
    files.append(str(p))

    plt.figure(figsize=(8, 5))
    plt.plot(hay["u"], hay["Ccrit_over_CR"], label="Hayward")
    plt.plot(bar["u"], bar["Ccrit_over_CR"], label="Bardeen")
    plt.xlabel(r"normalized deformation $u$")
    plt.ylabel(r"$C_{\rm crit}/(3\sqrt{3}\pi M^2)$")
    plt.title(r"Hayward vs Bardeen (full edge): normalized critical growth law")
    plt.legend()
    plt.tight_layout()
    p = outdir / "master_compare_edge_Ccrit_over_CR_overlay.png"
    plt.savefig(p, dpi=200)
    plt.close()
    files.append(str(p))

    plt.figure(figsize=(8, 5))
    plt.plot(common["u"], common["bardeen_over_hayward_Ccrit_over_CR"])
    plt.xlabel(r"normalized deformation $u$")
    plt.ylabel(r"Bardeen / Hayward")
    plt.title(r"Bardeen-to-Hayward ratio (full edge overlap)")
    plt.tight_layout()
    p = outdir / "master_compare_edge_ratio_bardeen_over_hayward.png"
    plt.savefig(p, dpi=200)
    plt.close()
    files.append(str(p))

    return files


# ============================================================
# Terminal summaries
# ============================================================
def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_small_table(df: pd.DataFrame, cols: list[str], n: int = 8) -> None:
    if df.empty:
        print("(empty)")
        return
    print(df[cols].head(n).to_string(index=False))


# ============================================================
# Main driver
# ============================================================
def run_master_driver(outdir: str | Path = ".", generate_plots: bool = True) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Universal clock summary
    clock_df = build_universal_clock_summary()
    clock_csv = outdir / "master_universal_clock_summary.csv"
    clock_df.to_csv(clock_csv, index=False)

    # 2. Load Hayward + Bardeen accumulated atlases
    hay = augment(load_hayward_full(Path(".")))
    bar = augment(load_bardeen_full(Path(".")))

    hay_csv = outdir / "master_hayward_atlas.csv"
    bar_csv = outdir / "master_bardeen_atlas.csv"
    hay.to_csv(hay_csv, index=False)
    bar.to_csv(bar_csv, index=False)

    # 3. Full-edge comparison
    common = build_common_grid(hay, bar, n=400)
    common_csv = outdir / "master_full_edge_common_grid.csv"
    common.to_csv(common_csv, index=False)

    combined = pd.concat([hay, bar], ignore_index=True).sort_values(["model", "u"]).reset_index(drop=True)
    combined_csv = outdir / "master_full_edge_combined_atlas.csv"
    combined.to_csv(combined_csv, index=False)

    # 4. Compact multimodel summary
    multimodel_rows: list[dict] = []
    for label, df in [("Hayward", hay), ("Bardeen", bar)]:
        multimodel_rows.append({
            "model": label,
            "M": float(df["M"].iloc[0]),
            "u_min": float(df["u"].min()),
            "u_max": float(df["u"].max()),
            "n_rows": int(len(df)),
            "Ccrit_over_CR_min": float(df["Ccrit_over_CR"].min()),
            "Ccrit_over_CR_max": float(df["Ccrit_over_CR"].max()),
            "tail_over_Ccrit_mean": float(df["tail_over_Ccrit"].dropna().mean()) if df["tail_over_Ccrit"].notna().any() else math.nan,
        })
    multimodel_csv = outdir / "master_multimodel_summary.csv"
    export_rows_csv(multimodel_rows, multimodel_csv)

    # 5. Manifest
    plot_files = make_plots(outdir, hay, bar, common) if generate_plots else []
    manifest_rows = [
        {"artifact": "universal_clock_summary", "path": str(clock_csv)},
        {"artifact": "hayward_atlas", "path": str(hay_csv)},
        {"artifact": "bardeen_atlas", "path": str(bar_csv)},
        {"artifact": "full_edge_common_grid", "path": str(common_csv)},
        {"artifact": "full_edge_combined_atlas", "path": str(combined_csv)},
        {"artifact": "multimodel_summary", "path": str(multimodel_csv)},
    ]
    for p in plot_files:
        manifest_rows.append({"artifact": "plot", "path": p})

    manifest_csv = outdir / "master_manifest.csv"
    export_rows_csv(manifest_rows, manifest_csv)

    # 6. Terminal display
    print_section("Universal Clock Summary")
    print_small_table(clock_df, ["Mass_Msun", "Coeff_C", "Invariant_I", "Log10_TauSat_Pl"], n=len(clock_df))

    print_section("Hayward Atlas Summary")
    print_small_table(hay, ["u", "x_star", "x_plus", "gap", "Ccrit_over_CR"], n=10)

    print_section("Bardeen Atlas Summary")
    print_small_table(bar, ["u", "x_star", "x_plus", "gap", "Ccrit_over_CR"], n=10)

    print_section("Full-Edge Comparison Summary")
    ratio_min = float(common["bardeen_over_hayward_Ccrit_over_CR"].min())
    ratio_max = float(common["bardeen_over_hayward_Ccrit_over_CR"].max())
    print(f"Common overlap: u in [{common['u'].min():.6f}, {common['u'].max():.6f}]")
    print(f"Bardeen/Hayward ratio of normalized Ccrit: min={ratio_min:.6f}, max={ratio_max:.6f}")
    print_small_table(
        common,
        [
            "u",
            "hayward_Ccrit_over_CR",
            "bardeen_Ccrit_over_CR",
            "bardeen_over_hayward_Ccrit_over_CR",
        ],
        n=10,
    )

    print_section("Artifacts written")
    for row in manifest_rows:
        print(row["path"])

    return {
        "clock_csv": str(clock_csv),
        "hayward_csv": str(hay_csv),
        "bardeen_csv": str(bar_csv),
        "common_csv": str(common_csv),
        "combined_csv": str(combined_csv),
        "multimodel_csv": str(multimodel_csv),
        "manifest_csv": str(manifest_csv),
        "plots": plot_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master reproducibility driver for the VIG project.")
    parser.add_argument("--outdir", type=str, default="master_results", help="Output directory for aggregated results.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_master_driver(outdir=args.outdir, generate_plots=not args.no_plots)
