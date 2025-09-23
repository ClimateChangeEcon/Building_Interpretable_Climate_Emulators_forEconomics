"""
===============================================================================
IPCC WGI v4 Regional + Gridded Warming Patterns at 2.5° × 2.5° for +1 °C GM
===============================================================================

PURPOSE
-------
This script computes and visualizes **local warming patterns** corresponding to a
**+1 °C increase in global-mean surface air temperature (GMT)** using CMIP pattern
files (e.g., CMIP5 “pattern scaling” datasets). It also aggregates the warming
over official **IPCC WGI v4 reference regions** (Land-only and Land+Land-Ocean)
and exports **β (beta) statistics** for selected regions needed in the paper.

WHAT THE SCRIPT PRODUCES
------------------------
For each model pattern file (HadGEM2-ES and MPI-ESM-LR by default), the script:

1) Interpolates local warming (ΔT_local = pattern × 1.0) to a **global 2.5° × 2.5°** grid
   in a consistent geographic coordinate system (Plate Carrée, lon ∈ [−180, 180]).

2) Creates **maps** (PNG):
   - 2.5° × 2.5° **masked** global map (only Land + Land-Ocean regions; oceans white)
   - **Land-only** WGI v4 regions colored by the regional mean warming
   - **Land + Land-Ocean** WGI v4 regions colored by the regional mean warming

3) Saves **grids** (CSV):
   - Full global 2.5° grid with local warming
   - Masked subset restricted to Land + Land-Ocean regions

4) Computes **β mean** (warming per +1 °C GM) for the selected regions
   **CNA, ECA, ARP, EEU, SAS**, and writes a per-model CSV.

5) Merges per-model β means into a **single comparative CSV** to support LaTeX tables.

All outputs are organized by model under:
    results_pattern_2p5_multi/<MODEL>/
and a combined CSV is written to:
    results_pattern_2p5_multi/region_stats_beta_plus1C_COMBINED.csv

WHY 2.5°?
---------
2.5° × 2.5° provides a tractable global grid that aligns with many macro datasets
and is sufficiently fine for regional means consistent with WGI polygons, while
keeping computations fast and simple.

INPUTS & EXPECTATIONS
---------------------
1) **Pattern NetCDF files** (CMIP5-like pattern-scaling data)
   - Default: 
       PATTERN_tas_ANN_HadGEM2-ES_rcp85.nc
       PATTERN_tas_ANN_MPI-ESM-LR_rcp85.nc
   - Must contain variables:
       * lat, lon  (1D or gridded coordinates; lon can be 0..360 or −180..180)
       * pattern   (local warming per 1 °C global warming, units: °C / °C)
     Optional variables (ignored for this script’s outputs): climatology, error
   - If lon ∈ [0, 360], we remap to lon ∈ [−180, 180].

2) **WGI v4 IPCC regions CSV** (v4 reference regions as lon|lat polygons)
   - Expected columns include: 
       "Continent / Ocean", "Surface", "Reference region name", "Acronym",
       "Vertex1", "Vertex2", … (plus unnamed columns for additional vertices)
   - The script reads **Land** and **Land-Ocean** rows to build polygons
     in geographic degrees (lon|lat). The **Ocean-only** regions are not used
     for masking nor for the two regional maps.

COORDINATE CONSISTENCY (IMPORTANT)
----------------------------------
- **Longitudes**: Everything is normalized to **[−180, 180]** using
  `((lon + 180) % 360) − 180`. This avoids dateline inconsistencies.
- **Projection**: All plotting uses Cartopy **PlateCarree()**, which expects
  lon/lat degrees. No arbitrary reprojection occurs.
- **Masking**: The 2.5° grid is filtered by the **union** of WGI Land + Land-Ocean
  polygons; points outside are dropped for the masked map. Regional means are
  computed by testing cell centers against region polygons.

REPRODUCIBILITY & ENVIRONMENT
-----------------------------
- Python ≥ 3.9 recommended.
- Suggested packages: numpy, pandas, xarray, scipy, shapely, matplotlib, cartopy
  (Cartopy may require GEOS/PROJ system libs depending on your platform.)
- Consider creating a virtual environment:
    python -m venv .venv
    source .venv/bin/activate         # (Windows: .venv\\Scripts\\activate)
    pip install numpy pandas xarray scipy shapely matplotlib cartopy

QUICK START
-----------
1) Place your input files in the working directory:
   - PATTERN_tas_ANN_HadGEM2-ES_rcp85.nc
   - PATTERN_tas_ANN_MPI-ESM-LR_rcp85.nc
   - ipcc_regions.csv  (WGI v4)

2) Run:
    python this_script.py

3) Inspect outputs under:
    results_pattern_2p5_multi/HadGEM2-ES/
    results_pattern_2p5_multi/MPI-ESM-LR/
   and the combined summary:
    results_pattern_2p5_multi/region_stats_beta_plus1C_COMBINED.csv

TROUBLESHOOTING
---------------
- **pattern variable missing**: ensure your NetCDF contains `pattern`.
- **Cartopy errors**: install system dependencies; on some systems:
    apt-get install libproj-dev proj-data proj-bin libgeos-dev
- **Regions not matching**: verify your WGI CSV is v4 and that the vertex
  columns are formatted as "lon|lat" pairs in degrees.

SCALABILITY / PERFORMANCE NOTES
-------------------------------
- Point-in-polygon checks at 2.5° are fast enough. For heavier workloads,
  consider STRtree (shapely) or rasterization of polygons.
- Interpolation uses `scipy.interpolate.griddata('cubic')` with nearest fill.
  For physically consistent regridding, consider `xesmf` (bilinear/conservative).

LICENSE / CITATION
------------------
If you use this code, please cite:
- Iturbide et al. (2020), ESSD 12, 2959–2970. IPCC WGI Reference Regions v4.
- The underlying CMIP datasets and data providers for the pattern files.

===============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import griddata
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.prepared import prep
from shapely.ops import unary_union

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ================================
# USER CONFIG
# ================================
PATTERN_FILES = [
    "PATTERN_tas_ANN_HadGEM2-ES_rcp85.nc",
    "PATTERN_tas_ANN_MPI-ESM-LR_rcp85.nc",
]
WGI_CSV = "ipcc_regions.csv"
OUTDIR_BASE = Path("results_pattern_2p5_multi")

GMT = 1.0             # global-mean warming increment (°C) -> β applies to +1 °C
RES = 2.5             # target grid resolution (degrees)

# Plotting
COLORMAP = "coolwarm" # diverging colormap as previously used
VMIN, VMAX = 0.25, 3.75 # colorbar limits for the +1 °C local warming maps
TITLE_FONTSIZE = 11

# Regions for β means (for Table 7)
SELECTED_REGIONS = ("CNA", "ECA", "ARP", "EEU", "SAS")


# ================================================================
# 1) Utilities
# ================================================================
def normalize_lon_to_180(lon_array: np.ndarray) -> np.ndarray:
    """Normalize longitude to [-180, 180]."""
    lon = np.asarray(lon_array, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0

def make_regular_grid(res=2.5):
    """
    Create center points of a global regular lat–lon grid at resolution `res` degrees.

    Centers:
      lon = [-180+res, -180+2*res, ..., 180-res]
      lat = [ -90+res,  -90+2*res, ...,  90-res]
    """
    lat_centers = np.arange(-90 + res,  90 + res,  res, dtype=float)
    lon_centers = np.arange(-180 + res, 180 + res, res, dtype=float)
    Lon, Lat = np.meshgrid(lon_centers, lat_centers)
    X_ = np.column_stack([Lat.ravel(), Lon.ravel()])
    return lat_centers, lon_centers, X_

def interp_to_grid(df, coord_cols=("lat","lon"), res=2.5, method="cubic"):
    """
    Interpolate all non-coordinate columns in `df` to a regular res×res grid using scipy.griddata.
    Fills NaNs left by 'cubic'/'linear' with 'nearest'.
    """
    df = df.dropna(subset=list(coord_cols)).copy()
    df[coord_cols[1]] = normalize_lon_to_180(df[coord_cols[1]].values)
    X = df.loc[:, coord_cols].values
    _, _, X_ = make_regular_grid(res=res)
    out = pd.DataFrame(X_, columns=list(coord_cols))
    data_cols = [c for c in df.columns if c not in coord_cols]
    for c in data_cols:
        y = df[c].values
        y_ = griddata(X, y, X_, method=method)
        if method in ("cubic","linear"):
            mask = np.isnan(y_)
            if np.any(mask):
                y_[mask] = griddata(X, y, X_[mask], method="nearest")
        out[c] = y_
    return out


# ==========================================
# 2) Pattern reading (+1°C local warming)
# ==========================================
def read_pattern_file(path_nc: str) -> pd.DataFrame:
    """
    Read NetCDF pattern file -> DataFrame with columns ['lat','lon','pattern'].
    """
    ds = xr.open_dataset(path_nc)
    if "pattern" not in ds.variables:
        raise ValueError(f"'pattern' variable not found in {path_nc}")
    df = ds.to_dask_dataframe().compute()
    keep = ["lat","lon","pattern"]
    df = df[keep].copy()
    df["lon"] = normalize_lon_to_180(df["lon"].values)
    return df

def compute_local_warming_for_gmt(df_pattern: pd.DataFrame, gmt=1.0) -> pd.DataFrame:
    """
    dT_local (°C) for +gmt °C = pattern (°C/°C) × gmt (°C).
    For GMT=1.0, dT_local == pattern.
    """
    out = df_pattern[["lat","lon"]].copy()
    out["dT_local"] = df_pattern["pattern"].values * float(gmt)
    return out


# ==============================================
# 3) WGI polygons
# ==============================================
def read_wgi_polygons(csv_path: str, surface_filter=("Land","Land-Ocean")):
    """
    Parse WGI v4 CSV; return dict: acronym -> {'surface':..., 'poly': shapely Polygon/MultiPolygon}.
    Only rows whose Surface is in `surface_filter` are returned.
    """
    regions = pd.read_csv(csv_path, sep=None, engine="python")
    out = {}
    for _, row in regions.iterrows():
        if row.get("Surface") not in surface_filter:
            continue
        verts = []
        for val in row.iloc[4:]:
            if isinstance(val, str) and "|" in val:
                lo, la = val.split("|")
                verts.append((float(lo), float(la)))
        if len(verts) < 3:
            continue
        poly = Polygon(verts)
        if not poly.is_valid:
            poly = poly.buffer(0)  # repair simple self-intersections
        out[row["Acronym"]] = {"surface": row["Surface"], "poly": poly}
    return out

def union_geometry(polydict: dict):
    """Return unary union of dict polygons."""
    geoms = []
    for rec in polydict.values():
        poly = rec["poly"]
        if isinstance(poly, MultiPolygon):
            geoms.extend(list(poly.geoms))
        else:
            geoms.append(poly)
    return unary_union(geoms) if geoms else None


# =========================================================
# 4) Plotting
# =========================================================
def plot_global_pattern_2p5_masked(df_grid_masked, title, outfile=None,
                                   vmin=VMIN, vmax=VMAX, cmap=COLORMAP):
    """
    2.5° map masked to Land + Land-Ocean (oceans white).
    """
    fig = plt.figure(figsize=(11,5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4, zorder=2)
    sc = ax.scatter(df_grid_masked["lon"], df_grid_masked["lat"],
                    c=df_grid_masked["dT_local"],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    s=12, transform=ccrs.PlateCarree(), zorder=1)
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.06)
    cbar.set_label("Local warming (°C for +1 °C)")
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        if str(outfile) == 'results_pattern_2p5_multi/MPI-ESM-LR/global_pattern_2p5_plus1C_land_plus.png':
            fig.savefig('figs_replication/figure_11_upper_left', dpi=300, bbox_inches="tight")
        if str(outfile) == 'results_pattern_2p5_multi/HadGEM2-ES/global_pattern_2p5_plus1C_land_plus.png':
            fig.savefig('figs_replication/figure_11_lower_left', dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_wgi_regions_colored_by_mean(pattern_grid, wgi_polys, title, outfile=None,
                                     vmin=VMIN, vmax=VMAX, cmap=COLORMAP):
    """
    Color each WGI region by the mean of dT_local over 2.5° cell centers inside it.
    """
    prepared = {k: prep(v["poly"]) for k, v in wgi_polys.items()}
    region_sum, region_count = {}, {}

    for (lat, lon, val) in pattern_grid[["lat","lon","dT_local"]].itertuples(index=False, name=None):
        pt = Point(lon, lat)
        for acr, poly in prepared.items():
            if poly.contains(pt) or poly.covers(pt):
                region_sum[acr] = region_sum.get(acr, 0.0) + float(val)
                region_count[acr] = region_count.get(acr, 0) + 1
                break

    region_mean = {acr: region_sum[acr]/region_count[acr] for acr in region_sum}

    fig = plt.figure(figsize=(11,5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4, zorder=2)
    cmap_ = plt.get_cmap(cmap); norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for acr, rec in wgi_polys.items():
        poly = rec["poly"]
        mean_val = region_mean.get(acr, np.nan)
        face = cmap_(norm(mean_val)) if np.isfinite(mean_val) else (0,0,0,0)
        polys = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
        for p in polys:
            ax.add_geometries([p], crs=ccrs.PlateCarree(),
                              facecolor=face, edgecolor="black",
                              linewidth=0.7, alpha=0.85, zorder=1)
        rep = poly.representative_point()
        ax.text(rep.x, rep.y, acr, ha="center", va="center",
                fontsize=7, color="k", transform=ccrs.PlateCarree(), zorder=3)

    sm = plt.cm.ScalarMappable(cmap=cmap_, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.06)
    cbar.set_label("Regional mean warming (°C for +1 °C)")
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print(outfile)
        if str(outfile) == 'results_pattern_2p5_multi/MPI-ESM-LR/regional_pattern_WGI_land_plus1C.png':
            fig.savefig('figs_replication/figure_11_upper_right', dpi=300, bbox_inches="tight")
        if str(outfile) == 'results_pattern_2p5_multi/HadGEM2-ES/regional_pattern_WGI_land_plus1C.png':
            fig.savefig('figs_replication/figure_11_lower_right', dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 5) Stats (β): mean only (no min/max in combined output)
# =========================================================
def compute_region_beta_means(pattern_grid: pd.DataFrame,
                              wgi_polys: dict,
                              selected=SELECTED_REGIONS,
                              model_name=None) -> pd.DataFrame:
    """
    Compute mean β (dT_local for +1 °C GM) over selected WGI regions.
    Returns DataFrame with columns: ['Acronym', f'beta_mean_{model_name}'].
    """
    prepared = {acr: prep(rec["poly"]) for acr, rec in wgi_polys.items() if acr in selected}
    values = {acr: [] for acr in selected}

    for (lat, lon, val) in pattern_grid[["lat","lon","dT_local"]].itertuples(index=False, name=None):
        pt = Point(lon, lat)
        for acr, poly in prepared.items():
            if poly.contains(pt) or poly.covers(pt):
                values[acr].append(float(val))
                break

    stats = []
    for acr in selected:
        arr = np.asarray(values[acr], dtype=float)
        mean_v = float(np.nanmean(arr)) if arr.size > 0 else np.nan
        stats.append({
            "Acronym": acr,
            f"beta_mean_{model_name}": mean_v,
        })
    return pd.DataFrame(stats)


# =========================================================
# 6) Main runner for a given model file
# =========================================================
def run_for_model(pattern_file: str) -> pd.DataFrame:
    """
    End-to-end run for a single pattern file. Writes maps, grids, per-model β means,
    and returns the β-mean DataFrame for combination across models.
    """
    model_name = Path(pattern_file).stem.replace("PATTERN_tas_ANN_", "").replace("_rcp85", "")
    outdir = OUTDIR_BASE / model_name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Running for {model_name} ===")

    # 1) Load patterns and compute +1 °C local warming
    df_pat = read_pattern_file(pattern_file)
    df_local = compute_local_warming_for_gmt(df_pat, gmt=GMT)  # ['lat','lon','dT_local']

    # 2) Interpolate to 2.5° global grid
    df_grid = interp_to_grid(df_local, coord_cols=("lat","lon"), res=RES, method="cubic")

    # 3) Read WGI regions
    wgi_land      = read_wgi_polygons(WGI_CSV, surface_filter=("Land",))
    wgi_land_plus = read_wgi_polygons(WGI_CSV, surface_filter=("Land","Land-Ocean"))

    # 4) Mask 2.5° grid to Land + Land-Ocean union (oceans white)
    land_plus_union = union_geometry(wgi_land_plus)
    land_plus_prep  = prep(land_plus_union)
    mask_inside = [
        land_plus_prep.contains(Point(lon,lat)) or land_plus_prep.covers(Point(lon,lat))
        for (lat,lon) in df_grid[["lat","lon"]].itertuples(index=False, name=None)
    ]
    df_grid_masked = df_grid.loc[mask_inside].reset_index(drop=True)

    # 5) Plots
    plot_global_pattern_2p5_masked(
        df_grid_masked,
        title=f"2.5° × 2.5° Pattern (+1 °C) — Land + Land–Ocean only ({model_name})",
        outfile=outdir/"global_pattern_2p5_plus1C_land_plus.png"
    )
    plot_wgi_regions_colored_by_mean(
        df_grid, wgi_land,
        title=f"Regional Pattern — WGI v4 (Land only) ({model_name})",
        outfile=outdir/"regional_pattern_WGI_land_plus1C.png"
    )
    plot_wgi_regions_colored_by_mean(
        df_grid, wgi_land_plus,
        title=f"Regional Pattern — WGI v4 (Land + Land-Ocean) ({model_name})",
        outfile=outdir/"regional_pattern_WGI_land_plus_landocean_plus1C.png"
    )

    # 6) Save grids
    df_grid.to_csv(outdir/"global_pattern_2p5_plus1C_FULL.csv", index=False)
    df_grid_masked.to_csv(outdir/"global_pattern_2p5_plus1C_LAND_PLUS.csv", index=False)

    # 7) β means (per-model CSV)
    df_stats = compute_region_beta_means(df_grid, wgi_land_plus, model_name=model_name)
    df_stats.to_csv(outdir/"region_stats_beta_plus1C.csv", index=False)

    print(f"Outputs written to {outdir.resolve()}")
    return df_stats


# =========================================================
# 7) Run for all models + COMBINED CSV
# =========================================================
if __name__ == "__main__":
    OUTDIR_BASE.mkdir(parents=True, exist_ok=True)

    # Run pipeline for each pattern file (keeps per-model outputs)
    all_stats = []
    for f in PATTERN_FILES:
        all_stats.append(run_for_model(f))

    # Merge per-model β means into a single wide CSV (no min/max)
    combined = all_stats[0]
    for df in all_stats[1:]:
        combined = pd.merge(combined, df, on="Acronym", how="outer")

    combined_path = OUTDIR_BASE / "region_stats_beta_plus1C_COMBINED.csv"
    combined.to_csv(combined_path, index=False)

    print(f"\nCombined β means written to: {combined_path.resolve()}")
    print(combined)
