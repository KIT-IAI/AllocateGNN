from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from sglib.generator import execution, stage
from . import tables


def _save(ctx, fig, name):
    path = stage.figures_root(ctx) / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig, path


def input_sizes(ctx):
    frame = tables.inputs(ctx)
    fig, ax = plt.subplots(figsize=(9, max(3, len(frame) * .35)))
    frame.set_index("region")[["grid_cells", "sources", "targets"]].plot.barh(ax=ax, logx=True,
        color=["#386b8e", "#d19a43", "#779665"])
    ax.set(xlabel="Count (log scale)", ylabel="Region", title=f"{ctx.country.upper()} inputs")
    ax.legend(["Grid cells", "Sources", "Targets"], frameon=False,
              loc="lower center", bbox_to_anchor=(.5, 1.13), ncol=3)
    fig.tight_layout()
    return _save(ctx, fig, "01_input_sizes.png")


def static_fields(ctx):
    bundle = execution._load_bundle(ctx.root)
    region = bundle.regions[0]
    grid = bundle.grids[region][0].to_crs(bundle.params["crs"]["working"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, component, title in zip(axes, ("uniform", "gpm", "public_activity"), ("Uniform", "GPM", "Public activity")):
        with np.load(ctx.root / "static" / component / f"{region}.npz", allow_pickle=False) as values:
            artist = ax.scatter(grid.geometry.x, grid.geometry.y, c=values["data"], s=2,
                                cmap="viridis", rasterized=True)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_axis_off()
        colorbar = fig.colorbar(artist, ax=ax, shrink=.65)
        colorbar.set_label(ctx.loaded.country_profile.units["demand"] if component != "public_activity" else "Activity mass")
    fig.suptitle(f"{ctx.country.upper()} · {region}")
    return _save(ctx, fig, "02_static_fields.png")


def candidate_coverage(ctx):
    frame = tables.candidates(ctx).groupby("family", sort=False)["fields"].sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    frame.plot.bar(ax=ax, color="#386b8e", rot=0)
    ax.set(xlabel="Family", ylabel="Materialized fields", title="Candidate coverage")
    fig.tight_layout()
    return _save(ctx, fig, "05_candidate_coverage.png")


def allocator_gates(ctx):
    counts = {}
    for name in ("idr_fixed", "idr_matched"):
        entries = json.loads((ctx.root / name / "index.json").read_text(encoding="utf-8"))["entries"]
        passed = sum(bool(e["g0_pass"] and e["g1_pass"]) for e in entries)
        counts[name] = (passed, len(entries) - passed)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Fixed IDR", "Matched IDR"]
    accepted, fallback = np.array(list(counts.values())).T
    ax.bar(labels, accepted, label="Gate passed", color="#779665")
    ax.bar(labels, fallback, bottom=accepted, label="VD fallback", color="#d19a43")
    ax.set(ylabel="Fields", title="Allocator decisions")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(ctx, fig, "06_allocator_gates.png")


# ---------------------------------------------------------------------------
# Physical-plausibility views: demand fields versus land use, and the
# Voronoi partition family (VD, fixed IDR, matched IDR).
# ---------------------------------------------------------------------------

import math

import geopandas as gpd
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LANDUSE_COLORS = {
    "residential": "#4664AA",
    "commercial": "#DF9B1B",
    "industrial": "#A11035",
    "agricultural": "#009682",
    "others": "#b3b3b3",
}
_ZERO_GREY = "#ececec"
_MAP_CRS = "EPSG:3857"


class _Raster:
    """Map per-cell values back onto the regular generation grid as an image."""

    def __init__(self, grid: gpd.GeoDataFrame, step: float):
        projected = grid.to_crs(_MAP_CRS)
        x = projected.geometry.x.to_numpy()
        y = projected.geometry.y.to_numpy()
        self.ix = np.rint((x - x.min()) / step).astype(int)
        self.iy = np.rint((y - y.min()) / step).astype(int)
        self.shape = (int(self.iy.max()) + 1, int(self.ix.max()) + 1)
        half = step / 2
        self.extent = (x.min() - half, x.max() + half, y.min() - half, y.max() + half)
        latitude = float(grid.to_crs("EPSG:4326").geometry.y.mean())
        self.map_metres_per_ground_metre = 1.0 / math.cos(math.radians(latitude))

    def image(self, values, fill=np.nan):
        image = np.full(self.shape, fill, dtype=float)
        image[self.iy, self.ix] = values
        return image

    def labels(self, values):
        image = np.full(self.shape, -1, dtype=int)
        image[self.iy, self.ix] = values
        return image

    def inside(self):
        return self.labels(np.zeros(len(self.ix), dtype=int)) >= 0

    def boundaries(self, labels):
        """RGBA layer marking cells whose right/upper neighbour carries another label."""
        edge = np.zeros(labels.shape, dtype=bool)
        inside = labels >= 0
        edge[:, :-1] |= inside[:, :-1] & inside[:, 1:] & (labels[:, :-1] != labels[:, 1:])
        edge[:-1, :] |= inside[:-1, :] & inside[1:, :] & (labels[:-1, :] != labels[1:, :])
        layer = np.zeros(labels.shape + (4,))
        layer[edge] = (0.1, 0.1, 0.1, 0.75)
        return layer

    def draw(self, ax, image, **kwargs):
        return ax.imshow(image, extent=self.extent, origin="lower", interpolation="nearest", **kwargs)

    def scalebar(self, ax, ground_km=10):
        x0, x1, y0, y1 = self.extent
        length = ground_km * 1000 * self.map_metres_per_ground_metre
        left = x0 + 0.04 * (x1 - x0)
        bottom = y0 + 0.05 * (y1 - y0)
        ax.plot([left, left + length], [bottom, bottom], color="black", linewidth=2, solid_capstyle="butt")
        ax.text(left + length / 2, bottom + 0.015 * (y1 - y0), f"{ground_km} km", ha="center", va="bottom", fontsize=7)


def _region_context(ctx, region):
    bundle = execution._load_bundle(ctx.root)
    region = region or bundle.regions[0]
    if region not in bundle.regions:
        raise ValueError(f"unknown region {region!r}; configured: {list(bundle.regions)}")
    grid, step = bundle.grids[region]
    return bundle, region, grid, float(step), _Raster(grid, float(step))


def _stations(bundle, region):
    return bundle.stations[region].to_crs(_MAP_CRS)


def _draw_stations(ax, stations, *, size=5, color="black"):
    ax.scatter(stations.geometry.x, stations.geometry.y, s=size, color=color, edgecolors="white", linewidths=0.3, zorder=5)


def _map_axes(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _candidate_entries(ctx, region):
    document = json.loads((ctx.root / "candidates/candidate_index.json").read_text(encoding="utf-8"))
    return [entry for entry in document["entries"] if entry["region"] == region]


def _default_seed(ctx):
    return int(ctx.loaded.values["training"]["seeds"][0])


def _field(ctx, region, label, seed):
    matches = [e for e in _candidate_entries(ctx, region) if e["label"] == label]
    if not matches:
        raise ValueError(f"no materialized field for label {label!r} in region {region!r}")
    seeded = [e for e in matches if e.get("seed") is not None]
    chosen = matches[0] if not seeded else next((e for e in seeded if int(e["seed"]) == int(seed)), None)
    if chosen is None:
        available = sorted({int(e["seed"]) for e in seeded})
        raise ValueError(f"{label!r} has no field for seed {seed}; available: {available}")
    with np.load(ctx.root / chosen["path"], allow_pickle=False) as archive:
        if not np.array_equal(archive["grid_row"], np.arange(len(archive["grid_row"]))):
            raise ValueError(f"{chosen['path']} is not in grid row order")
        return np.asarray(archive["data"], dtype=float), chosen.get("seed")


def _uniform(ctx, region):
    with np.load(ctx.root / "static/uniform" / f"{region}.npz", allow_pickle=False) as archive:
        return np.asarray(archive["data"], dtype=float)


def _gini(values):
    v = np.sort(np.asarray(values, dtype=float))
    if v.sum() <= 0:
        return float("nan")
    n = len(v)
    return float((2 * np.sum(np.arange(1, n + 1) * v) / (n * v.sum())) - (n + 1) / n)


def _top_share(values, fraction=0.10):
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    k = max(1, int(round(fraction * len(v))))
    return float(v[:k].sum() / v.sum()) if v.sum() > 0 else float("nan")


def _dominant_landuse(grid):
    columns = [c for c in grid.columns if c.startswith("lu_") and c.endswith("_prop")]
    names = [c.removeprefix("lu_").removesuffix("_prop") for c in columns]
    data = grid[columns].to_numpy(dtype=float)
    dominant = np.where(data.sum(axis=1) > 0, data.argmax(axis=1), -1)
    return names, dominant


def _load_fields(ctx, region, labels, seed):
    seed = _default_seed(ctx) if seed is None else int(seed)
    return {label: _field(ctx, region, label, seed) for label in labels}


def weight_fields(ctx, region=None, labels=("GPM", "MLP", "GNN"), seed=None, name=None):
    """2x2 map grid: dominant land use plus three demand fields relative to uniform.

    Each field panel shows log2(w / w_uniform): red cells hold more demand than
    an area-uniform spread, blue less; zero-support (Z) cells are light grey.
    Stations are overlaid on every panel. Pass exactly three labels so the grid
    stays 2x2; concentration statistics live in ``concentration_curves`` and
    ``concentration_shares``.
    """

    if len(labels) != 3:
        raise ValueError("weight_fields draws land use plus exactly three fields (2x2 grid)")
    bundle, region, grid, step, raster = _region_context(ctx, region)
    stations = _stations(bundle, region)
    uniform = _uniform(ctx, region)
    covered = uniform > 0
    names, dominant = _dominant_landuse(grid)
    fields = _load_fields(ctx, region, labels, seed)

    # Explicit layout from the raster aspect: two rows x two columns of maps,
    # one colourbar spanning both rows, the land-use legend under the grid.
    x0, x1, y0, y1 = raster.extent
    aspect = (x1 - x0) / (y1 - y0)
    map_h = 3.2
    map_w = map_h * aspect
    gap, margin, title_h, legend_h, cbar_w = 0.3, 0.3, 0.55, 0.45, 0.16
    fig_w = margin + 2 * map_w + gap + 0.35 + cbar_w + 1.0 + margin
    fig_h = 0.45 + 2 * (title_h + map_h) + gap + legend_h + 0.2
    fig = plt.figure(figsize=(fig_w, fig_h))

    def map_axes(index):
        row, column = divmod(index, 2)
        left = margin + column * (map_w + gap)
        bottom = fig_h - 0.45 - (row + 1) * (title_h + map_h) - row * gap
        return fig.add_axes([left / fig_w, bottom / fig_h, map_w / fig_w, map_h / fig_h])

    ax = map_axes(0)
    palette = [LANDUSE_COLORS.get(n, "#777777") for n in names]
    raster.draw(ax, np.where(raster.inside(), 0, np.nan), cmap=ListedColormap([_ZERO_GREY]))
    raster.draw(ax, raster.image(np.where(dominant >= 0, dominant, np.nan)),
                cmap=ListedColormap(palette), vmin=-0.5, vmax=len(names) - 0.5)
    _draw_stations(ax, stations)
    raster.scalebar(ax)
    _map_axes(ax, f"Dominant land use\n{len(stations)} stations")
    handles = [Patch(color=LANDUSE_COLORS.get(n, "#777777"), label=n) for n in names]
    handles.append(Line2D([], [], marker="o", linestyle="", color="black", markersize=3, label="station"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5 * (margin + 2 * map_w + gap) / fig_w, 0.05 / fig_h),
               fontsize=7, frameon=False, ncol=len(handles))

    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    ratio_cmap = plt.get_cmap("RdBu_r")
    artist = None
    for index, label in enumerate(labels, start=1):
        values, used_seed = fields[label]
        ratio = np.full(len(values), np.nan)
        ratio[covered] = np.log2(np.clip(values[covered] / uniform[covered], 2.0 ** -6, 2.0 ** 6))
        ax = map_axes(index)
        raster.draw(ax, np.where(raster.inside(), 0, np.nan), cmap=ListedColormap([_ZERO_GREY]))
        artist = raster.draw(ax, raster.image(ratio), cmap=ratio_cmap, norm=norm)
        _draw_stations(ax, stations)
        seed_text = f" · seed {used_seed}" if used_seed is not None else ""
        _map_axes(ax, f"{label}{seed_text}\nGini {_gini(values):.2f} · top 10 % cells hold {100 * _top_share(values):.0f} %")
    grid_bottom = fig_h - 0.45 - 2 * (title_h + map_h) - gap
    grid_h = 2 * map_h + title_h + gap
    cbar_left = margin + 2 * map_w + gap + 0.35
    cax = fig.add_axes([cbar_left / fig_w, (grid_bottom + 0.2 * grid_h) / fig_h, cbar_w / fig_w, 0.6 * grid_h / fig_h])
    colorbar = fig.colorbar(artist, cax=cax, ticks=[-3, -2, -1, 0, 1, 2, 3])
    colorbar.set_label("demand density relative to uniform")
    colorbar.ax.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])

    unit = ctx.loaded.country_profile.units["demand"]
    total = fields[labels[0]][0].sum()
    fig.suptitle(f"{ctx.country.upper()} · {region} · demand fields ({unit}, total {total:,.0f})", fontsize=11, y=1 - 0.15 / fig_h)
    return _save(ctx, fig, name or f"05_weight_fields_{region}.png")


_CONCENTRATION_LABELS = ("Uni", "GPM", "MLP", "GNN", "GNNpriorNP", "GNNpostNP", "GNNfusionNP")


def concentration_curves(ctx, region=None, labels=_CONCENTRATION_LABELS, seed=None, name=None):
    """Lorenz curves of the demand fields: cumulative demand against cells sorted densest first."""

    _, region, _, _, _ = _region_context(ctx, region)
    fields = _load_fields(ctx, region, labels, seed)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color=_ZERO_GREY, linewidth=1.5, label="uniform over all cells")
    for label in labels:
        values, _ = fields[label]
        v = np.sort(values)[::-1]
        share = np.concatenate([[0], np.cumsum(v) / v.sum()])
        ax.plot(np.linspace(0, 1, len(share)), share, linewidth=1.4, label=f"{label} (Gini {_gini(values):.2f})")
    ax.set(xlabel="share of cells, densest first", ylabel="cumulative share of demand", xlim=(0, 1), ylim=(0, 1))
    ax.set_title(f"{ctx.country.upper()} · {region} · concentration of demand", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(ctx, fig, name or f"05_concentration_curves_{region}.png")


def concentration_shares(ctx, region=None, labels=_CONCENTRATION_LABELS, seed=None, name=None):
    """Share of demand held by the densest 10 % of cells, one bar per field."""

    _, region, _, _, _ = _region_context(ctx, region)
    fields = _load_fields(ctx, region, labels, seed)
    shares = [100 * _top_share(fields[label][0]) for label in labels]
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(labels) + 1.4), constrained_layout=True)
    bars = ax.barh(list(labels), shares, color="#386b8e", height=0.6)
    ax.bar_label(bars, fmt="%.0f %%", padding=3, fontsize=7)
    ax.axvline(10, color=_ZERO_GREY, linewidth=1.5)
    ax.text(10.5, -0.45, "uniform", color="#888888", fontsize=7, ha="left", va="center")
    ax.set(xlabel="demand held by the densest 10 % of cells [%]", xlim=(0, max(shares) * 1.25))
    ax.invert_yaxis()
    ax.set_title(f"{ctx.country.upper()} · {region} · where the mass sits", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(ctx, fig, name or f"05_concentration_shares_{region}.png")


def _assignment(path):
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["assignment"], dtype=int)


def _index_entry(ctx, name, **match):
    entries = json.loads((ctx.root / name / "index.json").read_text(encoding="utf-8"))["entries"]
    for entry in entries:
        if all(str(entry.get(key)) == str(value) for key, value in match.items()):
            return entry
    raise ValueError(f"no {name} entry matching {match}")


def _gate_text(entry):
    g0 = "pass" if entry.get("g0_pass") else "fail"
    g1 = "pass" if entry.get("g1_pass") else "fail"
    tv = float(entry.get("tv_mass", float("nan")))
    budget = float(entry.get("transport_budget", float("nan")))
    mode = entry.get("selected_mode", "")
    reason = entry.get("fallback_reason") or ""
    text = f"G0 {g0} · G1 {g1} · TV {tv:.3f} of {budget:.2f} · {mode}"
    return text + (f"\n{reason}" if reason else "")


def partition_variants(ctx, region, candidate, seed):
    """Return the Voronoi assignment and the ordered variant records for one region.

    Each record is (title, station_of_cell, categories_of_cell, unit_label, note),
    with ``categories`` being the station index of every cell. CIVD is retired
    and deliberately absent from this comparison.
    """

    vd = _assignment(ctx.root / "static/assignments" / f"{region}.npz")
    variants = [("Voronoi (VD)", vd, vd, "stations", "")]
    fixed = _index_entry(ctx, "idr_fixed", region=region)
    fixed_assignment = _assignment(ctx.root / fixed["path"])
    variants.append(("IDR fixed (public activity)", fixed_assignment, fixed_assignment, "stations", _gate_text(fixed)))
    matched = _index_entry(ctx, "idr_matched", region=region, candidate=candidate, seed=seed)
    matched_assignment = _assignment(ctx.root / matched["path"])
    variants.append((f"IDR matched · {candidate} seed {seed}", matched_assignment, matched_assignment, "stations", _gate_text(matched)))
    return vd, variants


def _matched_seed(ctx, region, candidate, seed):
    entries = json.loads((ctx.root / "idr_matched/index.json").read_text(encoding="utf-8"))["entries"]
    seeds = sorted({int(e["seed"]) for e in entries if e["region"] == region and e["candidate"] == candidate})
    if not seeds:
        raise ValueError(f"no matched IDR field for candidate {candidate!r} in {region!r}")
    if seed is not None:
        return int(seed)
    default = _default_seed(ctx)
    return default if default in seeds else seeds[0]


_VARIANT_TITLES = {"Voronoi (VD)": "VD", "IDR fixed (public activity)": "IDR fixed"}


def _variant_title(title):
    if title in _VARIANT_TITLES:
        return _VARIANT_TITLES[title]
    return "IDR matched" if title.startswith("IDR matched") else title


def allocation_partitions(ctx, region=None, candidate="GNN", seed=None, name=None):
    """1x3 territory maps: Voronoi, fixed IDR and matched IDR for one region.

    Cells are coloured per station with dark boundaries and stations overlaid.
    Gate outcomes and reassignment shares are reported by ``partition_summary``
    and ``partition_departure``, not on the maps.
    """

    bundle, region, grid, step, raster = _region_context(ctx, region)
    stations = _stations(bundle, region)
    seed = _matched_seed(ctx, region, candidate, seed)
    vd, variants = partition_variants(ctx, region, candidate, seed)
    rng = np.random.default_rng(7)
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    permutation = rng.permutation(max(len(stations), 20))
    inside = raster.inside()

    fig, axes = plt.subplots(1, len(variants), figsize=(4.0 * len(variants), 4.4), constrained_layout=True)
    for ax, (title, _, categories, _, _) in zip(axes, variants):
        n_units = int(categories.max()) + 1
        colours = base[permutation[np.arange(n_units) % len(permutation)] % 20]
        labels = raster.labels(categories)
        rgba = np.zeros(labels.shape + (4,))
        rgba[labels >= 0] = colours[labels[labels >= 0]]
        rgba[..., 3] = np.where(labels >= 0, 0.55, 0.0)
        raster.draw(ax, np.where(inside, 0, np.nan), cmap=ListedColormap([_ZERO_GREY]))
        raster.draw(ax, rgba)
        raster.draw(ax, raster.boundaries(labels))
        _draw_stations(ax, stations, size=7)
        _map_axes(ax, _variant_title(title))
    fig.suptitle(f"{ctx.country.upper()} · {region} · allocation partitions ({len(stations)} stations)", fontsize=11)
    return _save(ctx, fig, name or f"06_partitions_{region}.png")


def partition_departure(ctx, region=None, candidate="GNN", seed=None, name=None):
    """Share of cells each variant assigns differently from the Voronoi partition."""

    _, region, _, _, _ = _region_context(ctx, region)
    seed = _matched_seed(ctx, region, candidate, seed)
    vd, variants = partition_variants(ctx, region, candidate, seed)
    names = [_variant_title(title) for title, *_ in variants]
    values = [100 * float(np.mean(station_of_cell != vd)) for _, station_of_cell, *_ in variants]
    fig, ax = plt.subplots(figsize=(5.2, 0.6 * len(names) + 1.4), constrained_layout=True)
    bars = ax.barh(names, values, color="#d19a43", height=0.6)
    ax.bar_label(bars, fmt="%.1f %%", padding=3, fontsize=7)
    ax.set(xlabel="cells assigned differently from Voronoi [%]", xlim=(0, max(5.0, max(values) * 1.3)))
    ax.invert_yaxis()
    ax.set_title(f"{ctx.country.upper()} · {region} · departure from Voronoi ({candidate} seed {seed})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(ctx, fig, name or f"06_partition_departure_{region}.png")
