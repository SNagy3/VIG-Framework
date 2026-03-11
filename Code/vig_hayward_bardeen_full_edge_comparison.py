from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd

PI = math.pi


def _require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_hayward_full(base: Path) -> pd.DataFrame:
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
        df = df[df["metric"].astype(str).str.lower().eq("hayward") | (df["ell"].astype(float) >= 0.0)].copy()
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


def load_bardeen_edge(base: Path) -> pd.DataFrame:
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


def make_plots(base: Path, hay: pd.DataFrame, bar: pd.DataFrame, common: pd.DataFrame) -> list[str]:
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
    p = base / "vig_compare_edge_xstar_overlay.png"
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
    p = base / "vig_compare_edge_xplus_overlay.png"
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
    p = base / "vig_compare_edge_gap_overlay.png"
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
    p = base / "vig_compare_edge_shape_function_overlay.png"
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
    p = base / "vig_compare_edge_Ccrit_over_CR_overlay.png"
    plt.savefig(p, dpi=200)
    plt.close()
    files.append(str(p))

    plt.figure(figsize=(8, 5))
    plt.plot(common["u"], common["bardeen_over_hayward_Ccrit_over_CR"])
    plt.xlabel(r"normalized deformation $u$")
    plt.ylabel(r"Bardeen / Hayward")
    plt.title(r"Bardeen-to-Hayward ratio (full edge overlap)")
    plt.tight_layout()
    p = base / "vig_compare_edge_ratio_bardeen_over_hayward.png"
    plt.savefig(p, dpi=200)
    plt.close()
    files.append(str(p))

    return files


def run_full_edge_comparison(outdir: str | Path = ".", common_n: int = 400) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hay = augment(load_hayward_full(outdir))
    bar = augment(load_bardeen_edge(outdir))

    combined = pd.concat([hay, bar], ignore_index=True)
    combined = combined.sort_values(["model", "u"]).reset_index(drop=True)
    common = build_common_grid(hay, bar, n=common_n)

    combined_csv = outdir / "vig_hayward_bardeen_full_edge_comparison.csv"
    common_csv = outdir / "vig_hayward_bardeen_full_edge_common_grid.csv"
    combined.to_csv(combined_csv, index=False)
    common.to_csv(common_csv, index=False)

    plots = make_plots(outdir, hay, bar, common)

    return {
        "combined_csv": str(combined_csv),
        "common_csv": str(common_csv),
        "plots": plots,
        "n_hayward": len(hay),
        "n_bardeen": len(bar),
        "common_n": len(common),
        "hayward_umax": float(hay["u"].max()),
        "bardeen_umax": float(bar["u"].max()),
    }


if __name__ == "__main__":
    result = run_full_edge_comparison(outdir=".")
    print("saved:")
    print(f"  {result['combined_csv']}")
    print(f"  {result['common_csv']}")
    for p in result["plots"]:
        print(f"  {p}")
    print(f"hayward rows = {result['n_hayward']}")
    print(f"bardeen rows = {result['n_bardeen']}")
    print(f"common grid  = {result['common_n']}")
    print(f"hayward u_max = {result['hayward_umax']:.6f}")
    print(f"bardeen u_max = {result['bardeen_umax']:.6f}")
