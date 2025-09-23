"""
================================================================================
South America — WGI-Consistent Damages & City Temperature Table (ΔT = 2.65−1.10)
================================================================================

Goal
----
Compute South-American damages under a global-mean warming increment of
ΔT = 1.55 °C (2.65 − 1.10), *strictly over WGI v4 reference regions* (Land + Land–Ocean),
while still visualizing 2.5° × 2.5° fields. Also generate a LaTeX table with
ERA5 (1991–2020) baseline temperatures and 2100 projections for the 15 specified cities.

Key requirements met
--------------------
1) **Strict WGI masking for computations**: all grid cells used for damages,
   statistics, and city values are FIRST filtered to the unary union of WGI v4
   SOUTH-AMERICA polygons (Land + Land–Ocean). This guarantees consistency with
   WGI regions even though the map is shown at 2.5° resolution.

2) **Coordinate consistency**:
   - All longitudes normalized to [-180, 180] with ((lon+180)%360)-180.
   - ERA5, CMIP patterns, WGI polygons, and our output grid share PlateCarree degrees.

3) **Outputs**:
   - Per-model SA grids with tas, β, tas_new, and damages (CSV).
   - Per-model WGI regional statistics (mean/min/max damages, mean β).
   - Combined regional statistics CSV (both models, side-by-side).
   - Two-panel damages figure (MPI-ESM-LR left, HadGEM2-ES right) with 15 specified cities.
   - **City temperature table** CSV and LaTeX (`sa_city_temps_table.tex`), columns:
       City, Country, ERA5 [°C], HadGEM2-ES 2100 [°C], MPI-ESM-LR 2100 [°C].

How to run
----------
1) Place inputs (or edit paths below):
   - ERA5_FILE = "Era5_temp/tas_climmean_era5_1991-2020_g025.nc"
   - CMIP patterns:
       "PATTERN_tas_ANN_MPI-ESM-LR_rcp85.nc"
       "PATTERN_tas_ANN_HadGEM2-ES_rcp85.nc"
   - WGI_CSV = "ipcc_regions.csv" (v4)
2) `pip install numpy pandas xarray scipy shapely matplotlib cartopy`
3) `python sa_wgi_damages_and_cities.py`
4) See outputs in `results_sa_wgi/`

Figure caption
--------------
Written to `results_sa_wgi/fig_caption.tex`:

\\caption{Damages, expressed as \( \tilde{D}(T_{2100}^{\text{z}})/\tilde{D}(T_{\text{Baseline}}^{\text{z}}) - 1 \),
assuming a global-mean warming of 2.65 °C since pre-industrial times ($\approx$ 1.55 °C since 2015) under the 4PR-X scenario.
Absolute temperatures stem from ERA5 (\(T_{\text{abs,c}}^{\text{z}}\)); pattern factors \( \beta^{\text{z}} \) are taken
from MPI-ESM-LR (left) or HadGEM2-ES (right). Red (blue) colors indicate an increase (decrease) in TFP due to global warming by 2100.}
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import griddata
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from shapely.prepared import prep

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ==============================================================================
# Configuration
# ==============================================================================
OUTDIR = Path("results_sa_wgi"); OUTDIR.mkdir(parents=True, exist_ok=True)

ERA5_FILE = "tas_climmean_era5_1991-2020_g025.nc"
CMIP_PATTERN_FILES = {
    "MPI-ESM-LR": "PATTERN_tas_ANN_MPI-ESM-LR_rcp85.nc",
    "HadGEM2-ES": "PATTERN_tas_ANN_HadGEM2-ES_rcp85.nc",
}
WGI_CSV = "ipcc_regions.csv"

GMT_INC = 2.65 - 1.10  # 1.55 °C
RES = 2.5

# Strict SOUTH-AMERICA bounding box (plotting convenience only; masking uses WGI polygons)
LAT_RANGE = (-60.0, 15.0)
LON_RANGE = (-100.0, -30.0)

# Plot settings
CMAP = "coolwarm"
VMAX = 0.25; VMIN = -VMAX
TITLE_FONTSIZE = 11

# 15 cities as in the provided table
SA_CITIES = [
    {"name": "São Paulo",     "country": "Brazil",    "lat": -23.5489, "lon": -46.6388},
    {"name": "Buenos Aires",  "country": "Argentina", "lat": -34.6037, "lon": -58.3816},
    {"name": "Rio de Janeiro","country": "Brazil",    "lat": -22.9068, "lon": -43.1729},
    {"name": "Lima",          "country": "Peru",      "lat": -12.0464, "lon": -77.0428},
    {"name": "Bogotá",        "country": "Colombia",  "lat":   4.7110, "lon": -74.0721},
    {"name": "Santiago",      "country": "Chile",     "lat": -33.4489, "lon": -70.6693},
    {"name": "Caracas",       "country": "Venezuela", "lat":  10.4806, "lon": -66.9036},
    {"name": "Quito",         "country": "Ecuador",   "lat":  -0.1807, "lon": -78.4678},
    {"name": "La Paz",        "country": "Bolivia",   "lat": -16.4897, "lon": -68.1193},
    {"name": "Brasília",      "country": "Brazil",    "lat": -15.7939, "lon": -47.8828},
    {"name": "Medellín",      "country": "Colombia",  "lat":   6.2442, "lon": -75.5812},
    {"name": "Guayaquil",     "country": "Ecuador",   "lat":  -2.1700, "lon": -79.9224},
    {"name": "Asunción",      "country": "Paraguay",  "lat": -25.2637, "lon": -57.5759},
    {"name": "Montevideo",    "country": "Uruguay",   "lat": -34.9011, "lon": -56.1645},
    {"name": "Curitiba",      "country": "Brazil",    "lat": -25.4297, "lon": -49.2719},
]


# ==============================================================================
# Utilities
# ==============================================================================
def normalize_lon_to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def make_regular_grid(lat_range, lon_range, res=2.5):
    lat = np.arange(lat_range[0], lat_range[1] + 1e-9, res)
    lon = np.arange(lon_range[0], lon_range[1] + 1e-9, res)
    Lon, Lat = np.meshgrid(lon, lat)
    X_ = np.column_stack([Lat.ravel(), Lon.ravel()])
    return lat, lon, X_


def interp_df_to_grid(df, lat_col="lat", lon_col="lon", res=2.5, lat_range=None, lon_range=None):
    """Interpolate df columns (except coords) onto a regular grid; nearest fill for holes."""
    if lat_range is None or lon_range is None:
        raise ValueError("lat_range and lon_range must be provided")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    df[lon_col] = normalize_lon_to_180(df[lon_col].values)

    X = df[[lat_col, lon_col]].values
    _, _, X_ = make_regular_grid(lat_range, lon_range, res=res)

    out = pd.DataFrame(X_, columns=[lat_col, lon_col])
    data_cols = [c for c in df.columns if c not in (lat_col, lon_col)]
    for c in data_cols:
        y = df[c].values
        y_ = griddata(X, y, X_, method="cubic")
        mask = np.isnan(y_)
        if np.any(mask):
            y_[mask] = griddata(X, y, X_[mask], method="nearest")
        out[c] = y_
    return out


# ==============================================================================
# Data loading
# ==============================================================================
def load_era5_to_df(path_nc, lat_range, lon_range):
    """Return DataFrame with ['lat','lon','tas'] (°C) on our 2.5° grid."""
    ds = xr.open_dataset(path_nc)
    var = ds["tas"].isel(time=0)
    units = (var.attrs.get("units") or "").lower()
    if var.lon.max() > 180:
        var = var.assign_coords(lon=normalize_lon_to_180(var.lon.values)).sortby("lon")
    var = var.sel(lat=slice(lat_range[0], lat_range[1]),
                  lon=slice(lon_range[0], lon_range[1]))
    df = var.to_dataframe().reset_index()
    if units.startswith("k"):
        df["tas"] = df["tas"] - 273.15
    df.rename(columns={"lat":"lat", "lon":"lon"}, inplace=True)
    return interp_df_to_grid(df[["lat","lon","tas"]], res=RES, lat_range=lat_range, lon_range=lon_range)


def load_pattern_to_df(path_nc, lat_range, lon_range):
    """Return DataFrame with ['lat','lon','pattern'] (°C per °C) on our grid."""
    ds = xr.open_dataset(path_nc)
    if "pattern" not in ds.variables:
        raise ValueError(f"'pattern' variable not found in {path_nc}")
    df = ds[["pattern"]].to_dataframe().reset_index()
    if "lon" not in df.columns or "lat" not in df.columns:
        raise ValueError("pattern file must provide lat/lon coordinates")
    df["lon"] = normalize_lon_to_180(df["lon"].values)
    df = df[(df["lat"]>=lat_range[0])&(df["lat"]<=lat_range[1])&(df["lon"]>=lon_range[0])&(df["lon"]<=lon_range[1])]
    return interp_df_to_grid(df[["lat","lon","pattern"]], res=RES, lat_range=lat_range, lon_range=lon_range)


# ==============================================================================
# WGI polygons (SOUTH-AMERICA only, Land + Land-Ocean)
# ==============================================================================
def read_wgi_sa_polygons(csv_path, surfaces=("Land","Land-Ocean")):
    regions = pd.read_csv(csv_path, sep=None, engine="python")
    out = {}
    for _, row in regions.iterrows():
        if row.get("Surface") not in surfaces:
            continue
        cont = str(row.get("Continent / Ocean","")).upper()
        if cont != "SOUTH-AMERICA":
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
            poly = poly.buffer(0)
        out[row["Acronym"]] = {"name": row["Reference region name"], "poly": poly}
    return out


def union_of_polygons(poly_dict):
    geoms = []
    for rec in poly_dict.values():
        g = rec["poly"]
        if isinstance(g, MultiPolygon):
            geoms.extend(list(g.geoms))
        else:
            geoms.append(g)
    return unary_union(geoms) if geoms else None


# ==============================================================================
# Damage function & computations (STRICT WGI cell selection)
# ==============================================================================
def damage_function(T, d=0.02, T_star=11.58, kappa_plus=0.00311, kappa_minus=0.00456):
    T = np.asarray(T, dtype=float)
    out = np.empty_like(T, dtype=float)
    mask = T >= T_star
    out[mask]  = (1-d) * np.exp(-kappa_plus  * (T[mask]-T_star)**2) + d
    out[~mask] = (1-d) * np.exp(-kappa_minus * (T[~mask]-T_star)**2) + d
    return out


def compute_damage_ratio_grid(df_era5, df_pattern, gmt_inc, sa_union, sa_polys):
    """
    Merge ERA5 and pattern, compute damages, and **assign WGI region** to each 2.5° cell.
    Keep only cells whose centers fall inside the SOUTH-AMERICA union (strict WGI).
    Return DataFrame: ['lat','lon','damages','beta','tas','tas_new','Region'].
    """
    df = pd.merge(df_era5, df_pattern, on=["lat","lon"], how="inner")
    df["beta"] = df["pattern"]
    df["tas_new"] = df["tas"] + df["beta"] * float(gmt_inc)

    D_new = damage_function(df["tas_new"].values)
    D_old = damage_function(df["tas"].values)
    df["damages"] = (D_new / D_old) - 1.0

    # Strict WGI masking
    union_p = prep(sa_union)
    inside = [union_p.contains(Point(lon, lat)) or union_p.covers(Point(lon, lat))
              for lat,lon in df[["lat","lon"]].to_numpy()]
    df = df.loc[inside].reset_index(drop=True)

    # Assign region acronym to each cell
    prepared = [(acr, prep(rec["poly"])) for acr, rec in sa_polys.items()]
    regions = []
    for (lat, lon) in df[["lat","lon"]].to_numpy():
        pt = Point(lon, lat)
        found = "NA"
        for acr, poly in prepared:
            if poly.contains(pt) or poly.covers(pt):
                found = acr
                break
        regions.append(found)
    df["Region"] = regions
    return df


# ==============================================================================
# Per-region statistics
# ==============================================================================
def regional_stats(df_grid, sa_polys):
    """
    Compute region stats using the explicit Region column (already WGI-consistent).
    Returns: ['Acronym','RefName','damages_mean','damages_min','damages_max','beta_mean'].
    """
    rows = []
    for acr, rec in sa_polys.items():
        sub = df_grid[df_grid["Region"] == acr]
        if sub.empty:
            rows.append({"Acronym": acr, "RefName": rec["name"],
                         "damages_mean": np.nan, "damages_min": np.nan, "damages_max": np.nan,
                         "beta_mean": np.nan})
        else:
            rows.append({"Acronym": acr, "RefName": rec["name"],
                         "damages_mean": float(sub["damages"].mean()),
                         "damages_min":  float(sub["damages"].min()),
                         "damages_max":  float(sub["damages"].max()),
                         "beta_mean":    float(sub["beta"].mean())})
    return pd.DataFrame(rows).sort_values("Acronym").reset_index(drop=True)


# ==============================================================================
# Cities (nearest WGI-consistent gridpoint)
# ==============================================================================
def nearest_city_values(df_grid, cities):
    """Nearest-gridpoint damages and temperatures for each city (grid is already WGI-filtered)."""
    pts = df_grid[["lat","lon"]].to_numpy()
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    out = []
    for c in cities:
        dist, idx = tree.query([c["lat"], c["lon"]], k=1)
        r = df_grid.iloc[idx]
        out.append({
            "City": c["name"], "Country": c["country"],
            "ERA5_C": float(r["tas"]),          # °C
            "T2100_C": float(r["tas_new"]),     # °C
            "beta": float(r["beta"]),
        })
    return pd.DataFrame(out)


def build_city_table_latex(city_era5: pd.DataFrame,
                           city_had: pd.DataFrame,
                           city_mpi: pd.DataFrame,
                           out_tex: Path):
    """
    Create a LaTeX table with columns:
      City, Country, ERA5 [°C], HadGEM2-ES 2100 [°C], MPI-ESM-LR 2100 [°C]
    The input data frames are aligned by City (case-sensitive).
    """
    # Merge ERA5 base (use either model's ERA5; we compute from the same grid)
    base = city_had[["City","Country"]].copy()
    base = base.merge(city_era5[["City","ERA5_C"]], on="City", how="left")
    base = base.merge(city_had[["City","T2100_C"]].rename(columns={"T2100_C":"Had_2100_C"}), on="City", how="left")
    base = base.merge(city_mpi[["City","T2100_C"]].rename(columns={"T2100_C":"MPI_2100_C"}), on="City", how="left")

    # Order rows by the original SA_CITIES list
    order = [c["name"] for c in SA_CITIES]
    base["__ord"] = base["City"].apply(lambda x: order.index(x) if x in order else 1e9)
    base = base.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)

    # Write CSV for convenience
    base.to_csv(out_tex.with_suffix(".csv"), index=False)

    # Build LaTeX source
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"    \centering")
    lines.append(r"    \renewcommand{\arraystretch}{1.2}")
    lines.append(r"    \begin{footnotesize}")
    lines.append(r"    \begin{tabular}{l l c c c}")
    lines.append(r"        \toprule")
    lines.append(r"        \textbf{City} & \textbf{Country} & \textbf{ERA5 [°C]} & \textbf{HadGEM2-ES 2100 [°C]} & \textbf{MPI-ESM-LR 2100 [°C]} \\")
    lines.append(r"        \midrule")
    for _, r in base.iterrows():
        lines.append(
            f"        {r['City']} & {r['Country']} & {r['ERA5_C']:.2f} & {r['Had_2100_C']:.2f} & {r['MPI_2100_C']:.2f} \\\\"
        )
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \end{footnotesize}")
    lines.append(r"    \caption{Temperature (in °C) “today’’, ERA5 1991–2020 mean, and as projected for 2100 in selected South-American cities.}")
    lines.append(r"    \label{tab:temperature_projection}")
    lines.append(r"\end{table}")
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    return base


# ==============================================================================
# Plot
# ==============================================================================
def plot_two_panel(models_to_df, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             constrained_layout=True)

    cmap = plt.get_cmap(CMAP)
    norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)

    order = ["MPI-ESM-LR", "HadGEM2-ES"]
    for ax, model in zip(axes, order):
        df = models_to_df[model]
        ax.set_extent([LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]])
        ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4, zorder=2)

        sc = ax.tricontourf(df["lon"], df["lat"], df["damages"],
                            levels=60, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

        # Cities
        for c in SA_CITIES:
            ax.plot(c["lon"], c["lat"], "o", ms=3.5, color="k", transform=ccrs.PlateCarree())
            ax.text(c["lon"]+0.5, c["lat"]+0.5, c["name"], fontsize=7,
                    transform=ccrs.PlateCarree())

        ax.set_title(f"Regional Damages in 2100 [{model}]", fontsize=11)

    # Shared colorbar labeled "Damages" (per your request)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Damages")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig('figure_12.png', dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================
def main():
    print(f"Using ΔT_GM = {GMT_INC:.2f} °C (2.65 − 1.10)")

    # ERA5 (°C) on our grid
    era5 = load_era5_to_df(ERA5_FILE, LAT_RANGE, LON_RANGE)

    # WGI SOUTH-AMERICA polygons and union
    sa_polys = read_wgi_sa_polygons(WGI_CSV)
    sa_union = union_of_polygons(sa_polys)
    if sa_union is None:
        raise RuntimeError("Could not build SOUTH-AMERICA WGI union.")

    # Per-model computations on STRICT WGI-filtered grid (and explicit Region tagging)
    models_to_df = {}
    per_model_stats = {}
    city_tables = {}

    for model, path in CMIP_PATTERN_FILES.items():
        print(f"\n--- {model} ---")
        pat = load_pattern_to_df(path, LAT_RANGE, LON_RANGE)
        grid = compute_damage_ratio_grid(era5, pat, GMT_INC, sa_union, sa_polys)

        # Save grid
        grid_path = OUTDIR / f"sa_grid_{model}.csv"
        grid.to_csv(grid_path, index=False)
        print(f"Wrote {grid_path}")

        # Regional stats
        stats_df = regional_stats(grid, sa_polys)
        stats_path = OUTDIR / f"sa_region_stats_{model}.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Wrote {stats_path}")

        # City table for this model (and ERA5 base)
        cities_df = nearest_city_values(grid, SA_CITIES)
        cities_path = OUTDIR / f"sa_cities_{model}.csv"
        cities_df.to_csv(cities_path, index=False)
        print(f"Wrote {cities_path}")

        models_to_df[model] = grid
        per_model_stats[model] = stats_df
        city_tables[model] = cities_df

    # Combine regional stats side-by-side (Had vs MPI)
    comb = per_model_stats["HadGEM2-ES"].merge(
        per_model_stats["MPI-ESM-LR"],
        on=["Acronym","RefName"],
        suffixes=("__HadGEM2-ES","__MPI-ESM-LR")
    )
    comb_path = OUTDIR / "sa_region_stats_COMBINED.csv"
    comb.to_csv(comb_path, index=False)
    print(f"Wrote {comb_path}")

    # Build city temperature LaTeX table (ERA5, Had 2100, MPI 2100)
    # ERA5 base is identical for both models (same baseline grid); use Had table for alignment.
    city_era5 = city_tables["HadGEM2-ES"][["City","ERA5_C"]].copy()
    city_had  = city_tables["HadGEM2-ES"][["City","Country","T2100_C"]].copy()
    city_mpi  = city_tables["MPI-ESM-LR"][["City","T2100_C"]].copy()

    tex_path = OUTDIR / "sa_city_temps_table.tex"
    city_tab = build_city_table_latex(city_era5, city_had, city_mpi, tex_path)
    print(f"Wrote {tex_path}")

    # Two-panel damages figure
    fig_path = OUTDIR / "sa_damages_two_panel.png"
    plot_two_panel(models_to_df, fig_path)
    print(f"Wrote {fig_path}")

    # Save the requested caption
    caption = r"""
\caption{Damages, expressed as \( \tilde{D}(T_{2100}^{\text{z}})/\tilde{D}(T_{\text{Baseline}}^{\text{z}}) - 1 \), assuming a global-mean warming of 2.65 °C since pre-industrial times ($\approx$ 1.55 °C since 2015) under the 4PR-X scenario. Absolute temperatures stem from ERA5 (\(T_{\text{abs,c}}^{\text{z}}\)); pattern factors \( \beta^{\text{z}} \) are taken from MPI-ESM-LR (left) or HadGEM2-ES (right). Red (blue) colors indicate an increase (decrease) in TFP due to global warming by 2100. }
""".strip()
    (OUTDIR / "fig_caption.tex").write_text(caption + "\n", encoding="utf-8")

    # Quick preview
    print("\nCity table (first 5 rows):")
    print(city_tab.head().to_string(index=False))


if __name__ == "__main__":
    main()
