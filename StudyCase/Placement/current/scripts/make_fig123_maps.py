"""Map figures 1-3 (plan section 8).

Fig. 1 problem map, Fig. 2 task map, Fig. 3 two-country coverage, with generated captions
(``figures_new/captions/*.tex`` input by the manuscript, ``*.md`` review copies). Geometry and fields
come from ``results/_backup/inputs`` (grid bundles, C/U/Z support, candidate fields,
substation dataset, r2 planning outputs, ONS ITL2 / ABS SA4 boundaries); screening
status, grid steps and costs come from ``frozen/<run_id>``.

Example-region rule (recorded in ``figures_new/fig123_sources.json``): the region
prespecified upstream as ``representative_region`` in casestudy/3_Experiment
(uk.toml: TLH3; au.toml: Sydney_North_Sydney_and_Hornsby), accepted only if it is
D2 Ref-eligible and every task is valid there. It is not chosen by LU/GNN improvement.
The illustrated neighbourhood is centred on the evaluation position nearest the
region's centroid. Maps are drawn in the national working CRS (true distances),
state the grid step and the aggregation radius, clip neighbourhoods to the region
boundary and draw no arrows; the real siting -> sizing dependency is shown by the
merged panel (b) of the task map (sized capacities drawn at the sites chosen by siting).
Layout (batch 12): every figure at most half a page, text at 7 pt or larger at 183 mm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.patches import ConnectionPatch
from matplotlib import patheffects
import numpy as np
import pandas as pd
from shapely.geometry import Point, MultiPoint
from shapely import voronoi_polygons

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper2_style  # noqa: E402
paper2_style.apply()
SPEC = {"uk": {"dir": "1_UK", "crs": "EPSG:27700", "key": "ITL3", "example": "TLH3", "demand": "Demand (MVA)",
               "regions": "data/datasets/2_derived/uk/bplus/regions.gpkg", "stations": "data/datasets/2_derived/uk/bplus/stations.gpkg",
               "unit": "MVA"},
        "au": {"dir": "2_AU", "crs": "EPSG:7856", "key": "SA3", "example": "Sydney_North_Sydney_and_Hornsby", "demand": "G_fy2024_mw",
               "regions": "data/datasets/2_derived/au/bplus/regions_sa3.gpkg", "stations": "data/datasets/2_derived/au/bplus/stations.gpkg",
               "unit": "MW"}}
X_MW = 300.0
UNIT_COST = 430_000.0  # GBP per MVA of reinforcement (c = GBP 430/kVA)
NAMES = {"TLH3": "Essex (TLH3)"}
CUZ_COLORS = ListedColormap(["#d9d9d9", "#f4e3b5", "#9ecae1"])  # Z, U, C
USED: dict[str, str] = {}


SIMPLIFY_M = {"region": 30.0, "national": 300.0}  # display-only tolerances (below 0.1 mm at the printed scale)


def simplified(gdf, level: str):
    """Display copy of a GeoDataFrame/GeoSeries with boundaries simplified at print resolution.

    Computation (centroids, clipping of neighbourhoods, frames) keeps the exact geometry; only drawn
    boundaries and fills use this copy, which keeps the vector PDFs small."""
    out = gdf.copy()
    geom = out.geometry.simplify(SIMPLIFY_M[level], preserve_topology=True)
    if isinstance(out, gpd.GeoDataFrame):
        out = out.set_geometry(geom)
    else:
        out = geom
    return out


def ordinal(value: float) -> str:
    """'2' -> '2nd', '99.8' -> '99.8th' (suffixes only for whole numbers; decimals take 'th')."""
    text = f"{value:g}"
    if float(value) != int(value):
        return text + "th"
    n = int(value)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return text + suffix


def number_word(n: int) -> str:
    """Small counts in running text as words (one to nine); larger counts stay numerals."""
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    return words[n] if 0 <= n < len(words) else f"{n:,}"


def signed(value: float, digits: int = 1) -> str:
    """Signed number with a typographic minus, e.g. -51.4 -> '−51.4', 2.0 -> '+2.0'."""
    text = f"{value:+.{digits}f}"
    return text.replace("-", "−")


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def use(backup: Path, path: Path) -> Path:
    USED[path.relative_to(backup).as_posix()] = sha(path)
    return path


def scalebar(ax, km=10, loc=(0.06, 0.06)):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x, y = x0 + loc[0] * (x1 - x0), y0 + loc[1] * (y1 - y0)
    ax.plot([x, x + km * 1000], [y, y], color="0.1", lw=1.5, solid_capstyle="butt")
    ax.text(x + km * 500, y + 0.015 * (y1 - y0), f"{km:g} km", ha="center", va="bottom", fontsize=6.5)


def clean(ax, title=None):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); paper2_style.map_axes(ax)
    if title:
        ax.set_title(title, loc="left")


class Case:
    def __init__(self, backup: Path, frozen: Path, cc: str):
        s = SPEC[cc]
        self.cc, self.s, self.backup = cc, s, backup
        base = backup / "inputs/base"
        support = pd.read_csv(use(backup, frozen / f"{cc}_region_support.csv"))
        self.support = support.set_index("region")
        stations = gpd.read_file(use(backup, base / s["stations"])).to_crs(s["crs"])
        stations["station_id"] = stations.station_id.astype(str)
        self.stations = stations.set_index("station_id")
        src = gpd.read_file(use(backup, base / s["regions"])).to_crs(s["crs"])
        self.sources = src
        rows = []
        for region in support.region:
            meta = json.loads(use(backup, base / f"data/datasets/2_derived/{cc}/grid_bplus/bundles/{region}/grid_metadata.json").read_text(encoding="utf-8"))
            part = src[src[s["key"]].astype(str).isin(list(map(str, meta["source_key_order"])))]
            rows.append({"region": region, "geometry": part.union_all()})
        self.regions = gpd.GeoDataFrame(rows, crs=s["crs"]).set_index("region")

    def region_stations(self, region):
        path = use(self.backup, self.backup / f"inputs/base/results/2_Generator/{self.s['dir']}/static/assignments/{region}.npz")
        with np.load(path, allow_pickle=False) as vd:
            ids, assignment = vd["station_id"].astype(str), vd["assignment"].copy()
        return self.stations.loc[ids], assignment

    def grid(self, region):
        p = use(self.backup, self.backup / f"inputs/base/data/datasets/2_derived/{self.cc}/grid_bplus/bundles/{region}/grid_points.parquet")
        g = gpd.read_parquet(p).to_crs(self.s["crs"])
        return np.column_stack([g.geometry.x, g.geometry.y])

    def cuz(self, region):
        p = use(self.backup, self.backup / f"inputs/base/data/datasets/2_derived/{self.cc}/features_bplus/extracted/{region}_cuz_support.npz")
        with np.load(p, allow_pickle=False) as z:
            return np.where(z["covered_mask"], 2, np.where(z["unknown_mask"], 1, 0))

    def field(self, region, label, seed):
        idx = json.loads(use(self.backup, self.backup / f"inputs/base/results/2_Generator/{self.s['dir']}/candidates/candidate_index.json").read_text(encoding="utf-8"))
        e = next(e for e in idx["entries"] if e["label"] == label and e["region"] == region and e.get("seed") == seed)
        with np.load(use(self.backup, self.backup / f"inputs/base/results/2_Generator/{self.s['dir']}" / e["path"]), allow_pickle=False) as z:
            return z["data"].astype(float)





WIDTH_IN = 183 / 25.4
COLUMN_IN = 88 / 25.4  # single-column width of the 5p layout (252 pt)
SYD_H = 1.62  # height (in) of the Sydney panel in the single-column coverage map  # Sydney inset in axes fractions of the Australian frame
METHOD_COLOURS = {"Uni": "#777777", "LU": "#E69F00", "GNN": "#0072B2", "Station register": "#343434"}
COST_COLOURS = ["#ffffd4", "#fed98e", "#fe9929", "#d95f0e", "#993404"]
COST_LABELS = ["<110", "110–120", "120–130", "130–140", "≥140"]


def export(fig, out, stem):
    fig.savefig(out / f"{stem}.pdf", bbox_inches=None)
    fig.savefig(out / f"{stem}.png", bbox_inches=None)
    plt.close(fig)


def neighbourhood(centre, radius_km, region_geom):
    return gpd.GeoSeries([Point(centre).buffer(radius_km * 1000, 256).intersection(region_geom)])


def example_position(case, frozen, region):
    cand = pd.read_parquet(use(case.backup, frozen / f"candidates/{case.cc}_candidates.parquet"),
                           filters=[("region", "==", region), ("radius_km", "==", 10.0), ("dc_mw", "==", 300.0)])
    centre = np.array(case.regions.loc[region, "geometry"].centroid.coords[0])
    uni = cand[cand.method == "Uni"]
    cid = int(uni.iloc[np.argmin(np.hypot(uni.x - centre[0], uni.y - centre[1]))].candidate_id)
    return cand, cand[cand.candidate_id == cid]


def dc_marker(ax, x, y):
    ax.plot([x], [y], marker="*", ms=8, color="#cb181d", mec="white", mew=0.6, zorder=8)


def map_frame(ax, geom, title, scale_km=None):
    x0, y0, x1, y1 = geom.bounds
    w, h = x1-x0, y1-y0
    ax.set_xlim(x0-.04*w, x1+.04*w)
    ax.set_ylim(y0-(.13 if scale_km else .04)*h, y1+.04*h)
    clean(ax, title)
    if scale_km:
        # A reserved strip below the geographic footprint holds the scale bar.
        xx, yy = x0+.18*w, y0-.095*h
        ax.plot([xx, xx+1000*scale_km], [yy, yy], color="0.2", lw=1.5, solid_capstyle="butt")
        ax.text(xx+500*scale_km, yy+.015*h, f"{scale_km:g} km", ha="center", va="bottom", fontsize=7.5)


FIG1_A_W = 1.10  # width (in) of panel (a) in Fig. 1: its one-line title must fit
FIG1_H = 3.72  # in; half-page limit of the author decision (batch 12)


def frame_ratio(geom, scale_strip):
    """Width/height of the frame that map_frame draws around geom (with or without the scale-bar strip)."""
    x0, y0, x1, y1 = geom.bounds
    return 1.08 * (x1 - x0) / ((1.17 if scale_strip else 1.08) * (y1 - y0))


def inch_axes(fig, x, y, w, h):
    """Axes placed in inches from the lower-left corner of the figure."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, y / H, w / W, h / H])


def cells(ax, xy, values, side, **kw):
    from matplotlib.collections import PatchCollection
    patches = [mpl.patches.Rectangle((x-side/2, y-side/2), side, side) for x, y in xy]
    pc = PatchCollection(patches, **kw); pc.set_array(np.asarray(values, float)); ax.add_collection(pc)
    return pc


def fig1(case, frozen, out, itl2):
    region = case.s['example']; xy = case.grid(region); cuz = case.cuz(region)
    st, _ = case.region_stations(region); geom = case.regions.loc[region, 'geometry']
    meta_path = use(case.backup, case.backup / f"inputs/base/data/datasets/2_derived/{case.cc}/grid_bplus/bundles/{region}/grid_metadata.json")
    side = float(json.loads(meta_path.read_text(encoding='utf-8'))['ground_neighbour_distance_m']['median'])
    cand_all, here = example_position(case, frozen, region)
    x0, y0 = float(here.x.iloc[0]), float(here.y.iloc[0]); circle = simplified(neighbourhood((x0,y0),10,geom),'region')
    fields = {"Uni": case.field(region,'Uni',None), "LU": case.field(region,'GPM',None), "GNN": case.field(region,'GNN',42)}
    positive = np.concatenate([v[v>0] for v in fields.values()])
    pct = (2, 99.8)
    norm = LogNorm(vmin=np.percentile(positive,pct[0]), vmax=np.percentile(positive,pct[1]))
    rows = {m: here[(here.method==m) & ((here.seed==42) if m=='GNN' else True)].iloc[0] for m in ('Uni','LU','GNN')}
    g, f = float(rows['Uni'].G), float(rows['Uni'].F)
    need = lambda value: max(0., value+X_MW-f)
    need_values = {'Station register':need(g), **{m:need(float(r.Ghat)) for m,r in rows.items()}}
    # mean fixed-location cost error over all candidate locations of the region (R = 10 km, X = 300 MW);
    # GNN averages the three seeds, as in every regional GNN result of the paper
    err=(cand_all.C_est-cand_all.C_true).abs()
    per_model=err.groupby([cand_all.method,cand_all.seed]).mean()
    mean_LE={m:float(per_model.loc[m].mean()) for m in ('Uni','LU','GNN')}
    n_positions={m:int((cand_all.method==m).sum()/max(1,cand_all[cand_all.method==m].seed.nunique())) for m in ('Uni','LU','GNN')}
    if len(set(n_positions.values()))!=1:raise ValueError('methods differ in candidate count')
    # half-page layout (batch 12): all positions in inches at the printed 183 mm width
    gb=simplified(itl2[~itl2.ITL221CD.str.startswith('TLN')],'national')
    country=gb.ITL221CD.str[:3].map(lambda c:{'TLL':'Wales','TLM':'Scotland'}.get(c,'England'))
    countries=gb.assign(country=country.values).dissolve('country')
    r_region=frame_ratio(geom,True);r_plain=frame_ratio(geom,False);r_gb=frame_ratio(countries.union_all(),True)
    # one grid for both rows: (d) (e) (f) of width wb, then a right column that holds the legend (top) and the colorbar
    # (bottom). Top row: (a) starts at the left edge of (d), (a)+(b) span (d)+(e), (c) has the edges of (f).
    title_h=.31;row_gap=.06;gap=.10;cb_w=.11;col_w=1.25;x_left=.04
    wb=(WIDTH_IN-2*x_left-2*gap-.12-col_w)/3;hb=wb/r_plain
    # (c) shows a window 8 km high whose width (whole km) matches the aspect of (f); the top row takes its height
    win_h_km=2*4.0;win_w_km=float(round(win_h_km*r_plain));ht=wb*win_h_km/win_w_km
    H=.04+hb+title_h+row_gap+ht+title_h+.04
    if H>3.9:raise ValueError(f'Fig. 1 taller than 3.9 in: {H:.2f}')
    fig=plt.figure(figsize=(WIDTH_IN,H))
    yb=.04;yt=yb+hb+title_h+row_gap;xb0=x_left
    xs=[xb0+k*(wb+gap) for k in range(3)];xcol=xs[2]+wb+.12
    # (a) wide enough for its one-line title, (b) takes the rest of (d)+(e); both maps fill their frames in height
    pos={'a':(xs[0],yt,FIG1_A_W,ht),'c':(xs[2],yt,wb,ht)}
    bx=xs[0]+FIG1_A_W+gap;pos['b']=(bx,yt,xs[1]+wb-bx,ht)
    rc=mpl.rc_context({'axes.titlesize':7.5});rc.__enter__()
    # (a) Great Britain as context: ITL2 regions of England, Wales and Scotland (Northern Ireland, TLN, is not part of
    # Great Britain), dissolved into the three countries; boundary file as for the coverage map
    a=inch_axes(fig,*pos['a'])
    countries.plot(ax=a,color='#f0f0f0',edgecolor='.45',lw=.3)
    national=simplified(case.regions,'national')
    national.plot(ax=a,color='#c6dbef',edgecolor='.55',lw=.15)
    national.loc[[region]].plot(ax=a,color='#08519c')
    map_frame(a,countries.union_all(),'(a) Great Britain',200)
    a.set_adjustable('datalim')
    ep=case.regions.loc[region,'geometry'].representative_point()
    a.annotate('Essex',xy=(ep.x,ep.y),xytext=(-16,16),textcoords='offset points',fontsize=7.2,ha='right',va='center',
               path_effects=[patheffects.withStroke(linewidth=2,foreground='white')],
               arrowprops=dict(arrowstyle='-',color='.3',lw=.4,shrinkA=1,shrinkB=1))
    # substation marker area proportional to the firm capacity in the frozen substation dataset
    firm=st['Firm Capacity (MVA)'].astype(float)
    missing=int(firm.isna().sum())
    FIRM_S=.30  # marker area (pt^2) per MVA of firm capacity
    st_size=np.where(firm.notna(),firm.fillna(0)*FIRM_S,1.5)
    print(f'fig1: {len(st)} substations in {region}; {missing} without firm-capacity data drawn with a fixed small marker'
          f'; firm capacity {firm.min():g}-{firm.max():g} MVA')
    b=inch_axes(fig,*pos['b'])
    b.scatter(xy[:,0],xy[:,1],c='#dfe8f1',s=.12,marker='s',lw=0,rasterized=True)
    b.scatter(st.geometry.x,st.geometry.y,s=st_size,c='k',lw=0)
    circle.boundary.plot(ax=b,color='#cb181d',lw=.8);dc_marker(b,x0,y0)
    zoom=500.*win_h_km;zoom_x=500.*win_w_km
    b.add_patch(mpl.patches.Rectangle((x0-zoom_x,y0-zoom),2*zoom_x,2*zoom,fill=False,ec='.25',lw=.6))
    need_title=lambda value:f'reinforcement requirement {value:.1f} MVA'
    map_frame(b,geom,f'(b) Reported-demand sum {g:.1f} MVA\n{need_title(need(g))}',10)
    b.set_adjustable('datalim')  # keep the widened frame, extend the map horizontally
    c=inch_axes(fig,*pos['c']);win=(abs(xy[:,0]-x0)<=zoom_x+side)&(abs(xy[:,1]-y0)<=zoom+side)
    gnn=fields['GNN'][win]
    cells(c,xy[win],np.where(gnn>0,gnn,np.nan),side,cmap='viridis',norm=norm,edgecolor='white',linewidth=.2)
    inwin=(abs(st.geometry.x-x0)<=zoom_x)&(abs(st.geometry.y-y0)<=zoom)
    c.scatter(st.geometry.x[inwin],st.geometry.y[inwin],s=np.maximum(st_size[inwin.to_numpy()],6),c='k',edgecolor='white',lw=.5,zorder=7)
    dc_marker(c,x0,y0)
    c.set_xlim(x0-zoom_x,x0+zoom_x);c.set_ylim(y0-zoom,y0+zoom)
    clean(c,f'(c) GNN allocation on\n{side:.0f} m raster cells')
    # stacked legend block: three map symbols, then the firm-capacity key on two lines
    legend_ax=inch_axes(fig,xcol,yt,WIDTH_IN-.02-xcol,ht+title_h);legend_ax.axis('off')
    firm_key=[5,15,45]
    kw=dict(frameon=False,fontsize=7.2,handlelength=1.1,handletextpad=.35,borderaxespad=0,borderpad=0,labelspacing=.5)
    leg=legend_ax.legend(handles=[mpl.lines.Line2D([],[],ls='none',marker='*',ms=7,c='#cb181d',label=f'{X_MW:.0f} MW data center'),
                                  mpl.lines.Line2D([],[],color='#cb181d',lw=.8,label='10 km neighborhood,\nclipped to the\nregion'),
                                  mpl.patches.Patch(fill=False,ec='.25',lw=.6,label='Window in (c)')],
                         loc='upper left',bbox_to_anchor=(0,.86),**kw)
    legend_ax.add_artist(leg)
    legend_ax.legend(handles=[mpl.lines.Line2D([],[],ls='none',marker='o',ms=np.sqrt(v*FIRM_S),c='k',label=f'{v:g}') for v in firm_key],
                     title='Substation firm\ncapacity (MVA)',title_fontsize=7.2,loc='upper left',bbox_to_anchor=(0,.30),ncol=3,
                     columnspacing=.5,alignment='left',**{**kw,'handlelength':.8,'handletextpad':.3})
    diff={}
    for k,(name,v) in enumerate(fields.items()):
        ax=inch_axes(fig,xs[k],yb,wb,hb)
        ax.scatter(xy[:,0],xy[:,1],c=np.where(v>0,v,np.nan),norm=norm,cmap='viridis',s=.22,marker='s',lw=0,rasterized=True)
        circle.boundary.plot(ax=ax,color='#cb181d',lw=.75);dc_marker(ax,x0,y0)
        ghat=float(rows[name].Ghat)
        # difference of the two displayed (0.1 MVA) values, kept as a fact; the panel titles show no difference
        diff[name]=round(round(ghat,1)-round(g,1),1)
        map_frame(ax,geom,f"({'def'[k]}) {name}: neighborhood demand {ghat:.1f} MVA\n{need_title(need(ghat))}")
    # the common source-area totals of (d)-(f), read from the frozen support table (stated in the caption)
    total=float(case.support.loc[region,'source_total'])
    if abs(total-float(case.support.loc[region,'station_demand_total']))>1e-6*total:raise ValueError('source and station totals differ')
    cbax=inch_axes(fig,xb0+3*wb+2*gap+.12,yb+.08,cb_w,hb-.16)
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=cbax,orientation='vertical',extend='both')
    cb.ax.tick_params(labelsize=7.5,length=2,pad=1,which='both')
    cb.set_label('Allocated demand per cell\n(MVA, log scale)',fontsize=7.5,labelpad=3)
    # the common input of (d)-(f): the regional total, in the right column level with the bottom-row titles (batch 13)
    total_label=f'Same source-area totals\nin (d)–(f): {total:,.1f} MVA'
    total_text=fig.text(xcol/WIDTH_IN,(yb+hb+title_h-.02)/H,total_label,fontsize=7.2,ha='left',va='top',linespacing=1.15)
    # every panel title must fit inside its frame (author rule); titles wrap to two lines where needed
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    col_px=(WIDTH_IN-.02-xcol)*fig.dpi
    if total_text.get_window_extent(renderer).width>col_px+1:raise ValueError('regional-total label wider than the right column')
    if total_text.get_window_extent(renderer).y0<cbax.get_window_extent(renderer).y1+2:raise ValueError('regional-total label overlaps the colorbar')
    for ax in fig.axes:
        t=ax.title
        if t.get_text() and t.get_window_extent(renderer).width>ax.get_window_extent(renderer).width+1:
            raise ValueError(f'panel title wider than its frame: {t.get_text()!r}')
    rc.__exit__(None,None,None)
    export(fig,out,'fig1_problem_map')
    return {'region':region,'region_name':NAMES[region],'n_sources':int(case.support.loc[region,'n_sources']),
            'regional_total_mva':total,
            'regional_total_source':'frozen uk_region_support.csv, row region=TLH3, field source_total (equal to station_demand_total); shown to 0.1 MVA',
            'window_km':win_h_km,'window_width_km':win_w_km,'window_height_km':win_h_km,'radius_km':10.0,'X_mw':X_MW,'unit_cost_gbp_m_per_mva':UNIT_COST/1e6,
            'difference_to_register_mva':diff,
            'station_marker':{'field':'Firm Capacity (MVA)','area_pt2_per_mva':FIRM_S,'key_mva':firm_key,
                              'n_stations':int(len(st)),'n_without_firm_capacity':missing},
            'context_map':'ITL2 January 2021 BFC without TLN, dissolved to England, Wales and Scotland',
            **{f'mean_fixed_location_cost_error_gbp_{m}':v for m,v in mean_LE.items()},'n_candidate_locations':n_positions['Uni'],
            'candidate_id':int(here.candidate_id.iloc[0]),'cell_side_m':side,'G':g,'F':f,
            'headroom':f-g,'reference_requirement':need(g),'requirements':need_values,
            'Ghat':{m:float(r.Ghat) for m,r in rows.items()},'T':float(rows['Uni']['T']),
            'reference_proxy_cost':need(g)*UNIT_COST+float(rows['Uni']['T']),
            'color_percentiles':list(pct),'demand_colour_limits':[float(norm.vmin),float(norm.vmax)]}


def cost_display_cells(cand,geom):
    xy=cand[['x','y']].to_numpy(float)
    if len(np.unique(xy,axis=0))!=len(xy):raise ValueError('Duplicate evaluation positions')
    polygons=list(voronoi_polygons(MultiPoint(xy),extend_to=geom.envelope,ordered=True).geoms)
    if len(polygons)!=len(cand):raise ValueError('Voronoi display cell count differs from candidate count')
    geometries=[];indices=[]
    for index,(polygon,point) in enumerate(zip(polygons,xy,strict=True)):
        if polygon.distance(Point(point))>1e-5:raise ValueError('Display cell/candidate mapping failed')
        clipped=polygon.intersection(geom)
        if not clipped.is_empty:geometries.append(clipped);indices.append(index)
    cost=cand.C_est.to_numpy(float)[indices]/1e6
    classes=np.digitize(cost,[110,120,130,140],right=False)
    return geometries,classes


def label_boundaries(xy,labels):
    """Internal boundaries of a raster partition: the Voronoi ridges between neighbouring cells whose labels differ.

    Exact for any cell-to-label assignment (no nearest-site assumption); segments are merged into lines."""
    from scipy.spatial import Voronoi
    from shapely.geometry import MultiLineString
    from shapely.ops import linemerge
    vor=Voronoi(xy);labels=np.asarray(labels)
    segments=[]
    for (i,j),ridge in zip(vor.ridge_points,vor.ridge_vertices):
        if labels[i]!=labels[j] and -1 not in ridge:
            segments.append(tuple(map(tuple,vor.vertices[ridge])))
    return gpd.GeoSeries([linemerge(MultiLineString(segments))])


def station_voronoi(st,geom):
    """Voronoi regions of the existing substations clipped to the region (display geometry)."""
    pts=np.column_stack([st.geometry.x,st.geometry.y])
    cells=voronoi_polygons(MultiPoint(pts),extend_to=geom.envelope)
    return gpd.GeoSeries([c.intersection(geom) for c in cells.geoms])


PARTITION_LINE=dict(color='.45',lw=.3)


def fig2(case,frozen,out):
    region=case.s['example'];xy=case.grid(region);st,assignment=case.region_stations(region)
    geom=case.regions.loc[region,'geometry'];exp=case.backup/f"inputs/release_r2/3_Experiment/{case.s['dir']}"
    with np.load(use(case.backup,exp/f'preflight/planning_pool/{region}.npz')) as z:pool=z['candidate_grid_rows']
    with np.load(use(case.backup,exp/f'planning/{region}/GNN/seed_42/selection.npz')) as z:catchment=z['grid_assignment'].copy()
    dec=pd.read_csv(use(case.backup,exp/f'planning/{region}/GNN/seed_42/decisions.csv'))
    dec=dec[dec.matching=='many_to_one'].sort_values('facility_ordinal');sites=xy[dec.candidate_grid_row.to_numpy(int)]
    cand,here=example_position(case,frozen,region);cand=cand[(cand.method=='GNN')&(cand.seed==42)].reset_index(drop=True)
    # the drawn Voronoi regions must be the partition the task uses (each cell to its nearest substation)
    from scipy.spatial import cKDTree
    nearest=cKDTree(np.column_stack([st.geometry.x,st.geometry.y])).query(xy)[1]
    if not np.array_equal(nearest,assignment):raise ValueError('station assignment is not the nearest-substation partition')
    shown=simplified(gpd.GeoSeries([geom],crs=case.s['crs']),'region').iloc[0]
    # one row of three panels (batch 12): (a) reconstruction, (b) siting and sizing merged, (c) connection;
    # the 488-site candidate pool is not drawn (its count is in the caption)
    # equal frames without a scale-bar strip; the scale bar of (a) sits in the empty south-east corner
    gap=.10;r_first=r_rest=frame_ratio(geom,False)
    hmap=(WIDTH_IN-.10-2*gap)/(3*r_rest)
    H=.20+hmap+.40+.24;fig=plt.figure(figsize=(WIDTH_IN,H))
    title_h=.20;ymap=H-title_h-hmap
    xs=[.05+k*(r_rest*hmap+gap) for k in range(3)]
    axes=[inch_axes(fig,x,ymap,r_rest*hmap,hmap) for x in xs]
    rc=mpl.rc_context({'axes.titlesize':8});rc.__enter__()
    outline=gpd.GeoSeries([shown],crs=case.s['crs'])
    for a in axes[:2]:outline.plot(ax=a,color='#f3f3f3',edgecolor='none')
    for a in axes:outline.boundary.plot(ax=a,color='.35',lw=.4)
    station_voronoi(st,shown).boundary.plot(ax=axes[0],**PARTITION_LINE)
    catchment_lines=label_boundaries(xy,catchment).intersection(shown)
    PEAK_S=.30;CAP_S=.30  # marker area (pt^2) per MVA
    axes[0].scatter(st.geometry.x,st.geometry.y,s=st[case.s['demand']]*PEAK_S,facecolor='none',edgecolor='k',lw=.45)
    map_frame(axes[0],geom,'(a) Peak-demand reconstruction')
    gx0,gy0,gx1,gy1=geom.bounds;sx,sy=gx1-.02*(gx1-gx0)-10000,gy0+.02*(gy1-gy0)
    axes[0].plot([sx,sx+10000],[sy,sy],color='.2',lw=1.5,solid_capstyle='butt')
    axes[0].text(sx+5000,sy+.02*(gy1-gy0),'10 km',ha='center',va='bottom',fontsize=7.5)
    catchment_lines.plot(ax=axes[1],**PARTITION_LINE)
    axes[1].scatter(sites[:,0],sites[:,1],s=dec.recommended_capacity*CAP_S,marker='^',facecolor='#c6dbef',edgecolor='#0072B2',lw=.5)
    map_frame(axes[1],geom,'(b) Substation siting and sizing')
    polys,classes=cost_display_cells(cand,shown)
    gpd.GeoSeries(polys,crs=case.s['crs']).plot(ax=axes[2],color=[COST_COLOURS[i] for i in classes],edgecolor='none',linewidth=0)
    axes[2].collections[-1].set_antialiased(False)
    outline.boundary.plot(ax=axes[2],color='.35',lw=.45)
    sel,orc=cand[cand.selected],cand[cand.oracle]
    # the Fig. 1 example: existing substations, the hypothetical data center and its clipped 10 km neighborhood
    axes[2].scatter(st.geometry.x,st.geometry.y,s=1.0,c='k',lw=0,zorder=4)
    dc_x,dc_y=float(here.x.iloc[0]),float(here.y.iloc[0])
    dc_circle=simplified(neighbourhood((dc_x,dc_y),10,geom),'region')
    dc_circle.boundary.plot(ax=axes[2],color='#cb181d',lw=.9,zorder=4.6)
    axes[2].plot([dc_x],[dc_y],marker='*',ms=9,color='#cb181d',mec='white',mew=.7,zorder=8)
    axes[2].scatter(sel.x,sel.y,s=15,facecolor='none',edgecolor='white',lw=1.6,zorder=5)
    axes[2].scatter(sel.x,sel.y,s=14,facecolor='none',edgecolor='#008767',lw=.8,zorder=6)
    axes[2].scatter(orc.x,orc.y,s=9,marker='x',c='white',lw=1.6,zorder=6)
    axes[2].scatter(orc.x,orc.y,s=9,marker='x',c='k',lw=.7,zorder=7)
    map_frame(axes[2],geom,'(c) Large-demand connection')
    # legend row 1: the size keys of (a) and (b) and the cost classes of (c), each under its panel
    y1=ymap-.37  # key block .33 in high, .04 in below the frames
    kw=dict(frameon=False,fontsize=7.5,title_fontsize=7.5,borderaxespad=0,borderpad=0,handletextpad=.3)
    for k,(vals,marker,colour,face,title,scale) in enumerate([([10,30,60],'o','k','none','Reported peak (MVA)',PEAK_S),
                                                              ([10,30,60],'^','#0072B2','#c6dbef','Sized capacity (MVA)',CAP_S)]):
        key=inch_axes(fig,xs[k+0],y1,(r_first if k==0 else r_rest)*hmap,.33);key.axis('off')
        key.legend(handles=[mpl.lines.Line2D([],[],ls='none',marker=marker,ms=np.sqrt(v*scale),mfc=face,mec=colour,mew=.5,label=f'{v:g}')
                            for v in vals],title=title,loc='center',ncol=3,columnspacing=1.2,handlelength=1.0,**kw)
    key=inch_axes(fig,xs[2],y1,r_rest*hmap,.33);key.axis('off')
    key.legend(handles=[mpl.patches.Patch(color=c,label=l) for c,l in zip(COST_COLOURS,COST_LABELS)],
               title='Estimated candidate cost (£m)',loc='center',ncol=5,handlelength=.8,columnspacing=.55,**kw)
    # legend row 2: line and point symbols shared across the panels
    row2=inch_axes(fig,.05,.03,WIDTH_IN-.10,.18);row2.axis('off')
    row2.legend(handles=[mpl.lines.Line2D([],[],**PARTITION_LINE,label='Voronoi region (a), catchment (b)'),
                         mpl.lines.Line2D([],[],ls='none',marker='o',ms=3.4,mfc='none',mec='#008767',mew=.75,label='Estimated shortlist'),
                         mpl.lines.Line2D([],[],ls='none',marker='x',ms=3.2,c='k',mew=.7,label='Reported-demand shortlist'),
                         mpl.lines.Line2D([],[],ls='none',marker='*',ms=6,c='#cb181d',label='Data center'),
                         mpl.lines.Line2D([],[],color='#cb181d',lw=.9,label='10 km neighborhood'),
                         mpl.lines.Line2D([],[],ls='none',marker='o',ms=1.6,c='k',label='Substation')],
                loc='center',ncol=6,columnspacing=1.0,handlelength=1.3,**{k:v for k,v in kw.items() if k!='title_fontsize'})
    rc.__exit__(None,None,None)
    export(fig,out,'fig2_task_map')
    share=float(np.mean(cand.C_true-cand['T']>1e-6))
    regret=float(orc.C_true.sum())
    regret=(float(sel.C_true.sum())-regret)/len(sel)
    return {'region':region,'sites':len(sites),'pool':len(pool),'grid_step_m':float(case.support.loc[region,'grid_step_m']),
            'selected_top1':len(sel),'overlap_with_register_top1':len(set(sel.candidate_id)&set(orc.candidate_id)),
            'selection_regret_gbp':regret,'cost_display':'candidate Voronoi cells clipped to region; no smoothing',
            'cost_bins_gbp_m':[110,120,130,140],'display_cells':len(polys),
            'share_needing_reinforcement':share,'n_evaluation_positions':len(cand),'radius_km':10.0,'X_mw':X_MW,
            'full_reinforcement_gbp_m':X_MW*UNIT_COST/1e6,
            'median_register_reinforcement_mva':float(np.median((cand.C_true-cand['T'])/430000)),
            'register_cost_range_gbp_m':[float(cand.C_true.min()/1e6),float(cand.C_true.max()/1e6)]}


def number_regions(ax,regions,numbers,fs=7.5):
    from matplotlib.transforms import Bbox
    occupied=[]
    for region in sorted(regions.index,key=lambda r:regions.loc[r,'geometry'].area):
        point=regions.loc[region,'geometry'].representative_point()
        label=str(numbers[region]);pixels_per_point=ax.figure.dpi/72
        anchor=ax.transData.transform((point.x,point.y))
        diameter=fs*(1.55 if len(label)==1 else 2.0)*pixels_per_point
        u=round(1.6*fs)
        offsets=[(0,0),(0,u),(u,0),(-u,0),(0,-u),(u,u),(-u,u),(u,-u),(-u,-u),
                 (0,2*u),(2*u,0),(-2*u,0),(0,-2*u),(2*u,u),(-2*u,u),(u,2*u),(-u,2*u),
                 (2*u,-u),(-2*u,-u),(u,-2*u),(-u,-2*u),(3*u,0),(-3*u,0),(0,3*u),(0,-3*u)]
        for dx,dy in offsets:
            centre=anchor+np.array([dx,dy])*pixels_per_point
            box=Bbox.from_bounds(centre[0]-diameter/2,centre[1]-diameter/2,diameter,diameter)
            inside=ax.bbox.contains(box.x0,box.y0) and ax.bbox.contains(box.x1,box.y1)
            if inside and not any(box.overlaps(b) for b in occupied):break
        else:
            raise ValueError(f'No collision-free region-label position: {region}')
        if dx or dy:
            ax.annotate('',xy=(point.x,point.y),xytext=(dx,dy),textcoords='offset points',
                        zorder=5,arrowprops=dict(arrowstyle='-',color='.4',lw=.35,shrinkA=5,shrinkB=2))
        ax.annotate(label,xy=(point.x,point.y),xytext=(dx,dy),textcoords='offset points',
                    fontsize=fs,ha='center',va='center',zorder=6,
                    path_effects=[patheffects.withStroke(linewidth=2.2,foreground='white')])
        occupied.append(box)


def coverage(ax,case,context,regions,title,scale,level='national'):
    simplified(context,level).boundary.plot(ax=ax,color='.82',lw=.22)
    shown=simplified(regions,level)
    ok=case.support.ref_eligible.astype(bool)
    shown.loc[[r for r in regions.index if ok[r]]].plot(ax=ax,color='#9ecae1',edgecolor='.45',lw=.35)
    bad=[r for r in regions.index if not ok[r]]
    if bad:shown.loc[bad].plot(ax=ax,color='#deebf7',edgecolor='.6',lw=.35,hatch='\\\\')
    pts=pd.concat([case.region_stations(r)[0] for r in regions.index])
    ax.scatter(pts.geometry.x,pts.geometry.y,s=.6,c='k',lw=0)
    map_frame(ax,regions.union_all(),title,scale)


def fig3(cases,frozen,out,itl2,sa4_all):
    led=pd.read_csv(use(cases['uk'].backup,frozen/'ledger/claim_ledger.csv'),dtype={'value':str}).set_index('claim_id')
    val=lambda k:int(float(led.loc[k,'value']))
    names_uk=dict(zip(itl2.ITL221CD,itl2.ITL221NM));nsw=sa4_all[sa4_all.STE_NAME21=='New South Wales']
    numbers={};keys={}
    for cc in ('uk','au'):
        case=cases[cc];order=case.regions.geometry.apply(lambda g:-g.representative_point().y).sort_values().index
        numbers[cc]={r:i for i,r in enumerate(order,1)}
        keys['GB' if cc=='uk' else 'AU']=[{'number':i,'region':r,'name':names_uk.get(r,r) if cc=='uk' else r.replace('_',' ').replace(' exc Newcastle','')}
                                          for r,i in numbers[cc].items()]
    title=lambda lab,name:f"{name}\n{val(f'DATA.{lab}.n_regions')} regions, {val(f'DATA.{lab}.n_stations'):,} substations"
    # single-column layout (batch 12, 88 mm): Britain as the tall left panel; right column with Australia on top and
    # Sydney enlarged below it (rectangle in (b)); all text at 7-7.5 pt
    r_uk=frame_ratio(cases['uk'].regions.union_all(),True);r_au=frame_ratio(cases['au'].regions.union_all(),True)
    syd=cases['au'].regions[[r.startswith('Sydney_') for r in cases['au'].regions.index]]
    r_syd=frame_ratio(syd.union_all(),True)
    gap=.08;title_h=.30;leg_h=.30;w_col=1.36
    w_uk=COLUMN_IN-.04-gap-w_col;h=w_uk/r_uk
    H=leg_h+h+title_h+.02
    fig=plt.figure(figsize=(COLUMN_IN,H))
    rc=mpl.rc_context({'axes.titlesize':7.5});rc.__enter__()
    ukax=inch_axes(fig,.02,leg_h,w_uk,h);coverage(ukax,cases['uk'],itl2,cases['uk'].regions,title('GB','(a) Britain'),100)
    # right column: Sydney (c) gets SYD_H of the height, Australia (b) the rest below its two-line title
    xr=.02+w_uk+gap;h_syd=SYD_H;w_syd=r_syd*h_syd
    h_au=h-h_syd-.30;w_au=min(w_col,r_au*h_au);h_au=w_au/r_au;y_au=leg_h+h-h_au
    auax=inch_axes(fig,xr,y_au,w_col,h_au)  # frame as wide as the column so the title fits; map extends east-west
    coverage(auax,cases['au'],nsw,cases['au'].regions,title('AU','(b) Australia, Ausgrid'),20)
    auax.set_adjustable('datalim')
    inset=inch_axes(fig,xr+(w_col-w_syd)/2,leg_h,w_syd,h_syd)
    coverage(inset,cases['au'],nsw,syd,'(c) Sydney',5,'region')
    sx0,sy0,sx1,sy1=syd.union_all().bounds
    auax.add_patch(mpl.patches.Rectangle((sx0,sy0),sx1-sx0,sy1-sy0,fill=False,ec='.25',lw=.6,zorder=7))
    leg=inch_axes(fig,.02,.0,COLUMN_IN-.04,leg_h-.04);leg.axis('off')
    leg.legend(handles=[mpl.patches.Patch(color='#9ecae1',label='Ref reliable here'),
                        mpl.patches.Patch(facecolor='#deebf7',edgecolor='.6',hatch='\\\\',label='Other analysis regions'),
                        mpl.lines.Line2D([],[],ls='none',marker='o',ms=2.5,c='k',label='Substations')],
               loc='center',ncol=2,frameon=False,fontsize=7.5,handlelength=1.3,columnspacing=1.0,handletextpad=.4,
               borderaxespad=0,borderpad=0,labelspacing=.3)
    fig.canvas.draw()
    number_regions(ukax,cases['uk'].regions,numbers['uk'],7)
    number_regions(auax,cases['au'].regions[[not r.startswith('Sydney_') for r in cases['au'].regions.index]],numbers['au'],7)
    number_regions(inset,syd,numbers['au'],7)
    rc.__exit__(None,None,None)
    export(fig,out,'fig3_coverage')
    return {c:{'n_regions':val(f'DATA.{c}.n_regions'),'n_stations':val(f'DATA.{c}.n_stations'),'eligible':val(f'REF.{c}.n_eligible')}
            for c in ('GB','AU')}|{'region_key':keys}


def write_region_table(out,keys,source):
    """Numbered-region key of the coverage map (fig:coverage) as a table (``figures_new/tab_regions.tex``), one block
    per country, from the same region key that numbers the map; each printed number is bound to that key for the
    number checker."""
    tex=lambda s:s.replace('&','\\&').replace('%','\\%').replace('_','\\_')
    fact='F:fig123_sources.json:fig3.region_key'
    lines=[]
    for cc,title in (('GB','Britain'),('AU','Australia')):
        if cc=='AU':lines.append('    \\midrule')
        lines.append(f'    \\multicolumn{{3}}{{@{{}}l}}{{{title}}} \\\\')
        for i,r in enumerate(keys[cc]):
            code=r['region'] if cc=='GB' and r['region'].startswith('TL') else ''
            lines.append(f"    {r['number']} & {tex(r['name'])} & {code} \\\\ %@ {fact}.{cc}.{i}.number")
    body='\n'.join([
        f"% generated by scripts/make_fig123_maps.py from the frozen run; do not edit by hand. Source: {source}",
        '\\begin{table}[!ht]','  \\centering',
        '  \\caption{Numbered analysis regions of Fig.~\\ref{fig:coverage}.',
        '  British regions are built from \\acrfull{itl2} regions and are listed with their \\acrshort{itl2} code.',
        "  London combines Greater London's five \\acrshort{itl2} regions and has no single code.",
        '  Australian regions are \\acrfull{sa4} regions in the Ausgrid network area of New South Wales.}',
        '  \\label{tab:regions}','  \\footnotesize',
        '  \\begin{tabular}{@{}rll@{}}','    \\toprule',
        '    No. & Region & \\acrshort{itl2} code \\\\','    \\midrule',
        *lines,'    \\bottomrule','  \\end{tabular}','\\end{table}',''])
    (Path(out)/'tab_regions.tex').write_text(body,encoding='utf-8')


def main():
    here=Path(__file__).resolve();ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--backup',type=Path,default=here.parents[2]/'results/_backup')
    ap.add_argument('--out',type=Path,default=here.parents[1]/'figures_new');args=ap.parse_args()
    backup=args.backup.resolve();frozen=backup/'frozen'/args.run_id;args.out.mkdir(parents=True,exist_ok=True)
    cases={cc:Case(backup,frozen,cc) for cc in ('uk','au')}
    for cc,case in cases.items():
        if not bool(case.support.loc[case.s['example'],'ref_eligible']):raise ValueError('Prespecified example is not eligible')
    itl2=gpd.read_file(use(backup,backup/'inputs/boundaries/uk_itl2_jan2021_bfc_v3.geojson')).to_crs('EPSG:27700')
    sa4=gpd.read_file(f"zip://{use(backup,backup/'inputs/boundaries/au_sa4_2021_gda2020_shp.zip')}").dropna(subset=['geometry']).to_crs('EPSG:7856')
    facts={'fig1':fig1(cases['uk'],frozen,args.out,itl2),'fig2':fig2(cases['uk'],frozen,args.out)}
    facts['fig3']=fig3(cases,frozen,args.out,itl2,sa4)
    f1=facts['fig1'];f2=facts['fig2'];f3=facts['fig3']
    src=f"frozen/{args.run_id} (support table, candidates, ledger) and inputs/base; facts in figures_new/fig123_sources.json"
    req=f1['requirements']
    le=lambda m:f1[f'mean_fixed_location_cost_error_gbp_{m}']/1e6
    ACR={'Uni':'\\acrshort{uni}','LU':'\\acrshort{lu}','GNN':'\\acrshort{gnn}'}  # glossaries short forms in captions
    if f1['station_marker']['n_without_firm_capacity']:raise ValueError('caption wording assumes firm capacity for every substation')
    paper2_style.write_caption(args.out,'fig1_problem_map','fig:problem',
        "The same source-area totals, allocated in different ways, give the same location different reinforcement requirements. "
        f"{f1['region_name']} is the representative region fixed in the protocol. "
        "(a)~Great Britain: British analysis regions light blue, Essex dark blue. "
        "(b)~Substations (marker area scaled to firm capacity), a hypothetical "
        f"{f1['X_mw']:.0f}~MW data center (red star) at the candidate location nearest the region centroid and its "
        f"{f1['radius_km']:.0f}~km neighborhood (red), over which demand and firm capacity are summed. "
        f"(c)~\\Acrfull{{gnn}}-based allocation from seed 42 in the {f1['window_width_km']:.0f}~km$\\times${f1['window_height_km']:.0f}~km "
        f"window of (b), on raster cells {f1['cell_side_m']:.0f}~m apart. "
        "(d, e, f)~Demand per raster cell in MVA (shared logarithmic scale) under \\acrfull{uni}, \\acrfull{lu} and \\acrshort{gnn}-based allocation, "
        f"whose source-area totals sum to {f1['regional_total_mva']:,.1f}~MVA. "
        "Panel titles give the neighborhood demand and the reinforcement requirement, the incoming load minus the headroom "
        "that this demand leaves.", src)
    if f2['share_needing_reinforcement']!=1.0:raise ValueError('caption wording assumes every position needs reinforcement')
    if f2['overlap_with_register_top1']!=1:raise ValueError('caption wording assumes a single shared location')
    paper2_style.write_caption(args.out,'fig2_task_map','fig:tasks',
        "The same demand allocation enters four related evaluations. "
        f"{f1['region_name']}, \\acrfull{{gnn}}-based allocation from seed 42. "
        "(a)~Peak-demand reconstruction: demand summed over each substation's Voronoi region is compared with its reported peak demand. "
        f"(b)~Substation siting and sizing: {f2['sites']} of {f2['pool']} candidate sites (not shown) are selected, each serving a "
        "catchment, and sized from its catchment demand. "
        f"(c)~Large-demand connection: estimated costs of connecting {f2['X_mw']:.0f}~MW at {f2['n_evaluation_positions']:,} "
        f"candidate locations using demand within {f2['radius_km']:.0f}~km. "
        "Each cost is displayed on the candidate location's Voronoi polygon. "
        f"The estimated and reported-demand shortlists are the {f2['selected_top1']} cheapest locations (top 1~\\%) under each cost and share {f2['overlap_with_register_top1']}. Choosing the estimated shortlist "
        f"costs on average \\pounds{f2['selection_regret_gbp']/1e6:.2f}m more per location under reported demand (selection regret). "
        "The data center and its neighborhood are those of Fig.~\\ref{fig:problem}. Under reported demand every location needs "
        f"reinforcement (\\pounds{f2['register_cost_range_gbp_m'][0]:.0f}m to "
        f"\\pounds{f2['register_cost_range_gbp_m'][1]:.0f}m).", src)
    keys=f3['region_key']
    paper2_style.write_caption(args.out,'fig3_coverage','fig:coverage',
        "Analysis regions of the two case studies: (a)~Britain, (b)~Australia and (c)~Sydney, the rectangle in (b). "
        "The \\acrfull{ref} is reliable in the blue regions (Sec.~\\ref{sec:eligibility}). "
        "Boundaries: ONS ITL2 January 2021 BFC under OGL v3.0 (contains OS data, Crown copyright 2021) and ABS ASGS Edition~3 SA4 2021 "
        "under CC BY 4.0.", src)
    write_region_table(args.out,keys,src)
    boundaries=json.loads(use(backup,backup/'inputs/boundaries/provenance.json').read_text(encoding='utf-8'))
    for cc,case in cases.items():
        config_root=backup/f"inputs/release_r2/3_Experiment/{case.s['dir']}/setup/config"
        for cfg in config_root.rglob(f'{cc}.toml'):
            if '/3_Experiment/' in cfg.as_posix():use(backup,cfg)
    record={'run_id':args.run_id,'script_sha256':sha(here),'facts':facts,'example_regions':{cc:c.s['example'] for cc,c in cases.items()},
            'example_rule':'upstream representative_region; D2 eligible; not selected by improvement',
            'neighbourhood_centre_rule':'evaluation position nearest the region centroid',
            'display_width_mm':183,'boundaries':boundaries,'inputs_sha256':dict(sorted(USED.items()))}
    (args.out/'fig123_sources.json').write_text(json.dumps(record,indent=2,ensure_ascii=False,default=float),encoding='utf-8')
    print(json.dumps(facts,ensure_ascii=True,default=float))


if __name__=='__main__':
    main()
