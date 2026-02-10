import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pydeck as pdk
from matplotlib import colormaps


def load_edges(path: Path) -> gpd.GeoDataFrame:
    """Load edges GeoPackage and extract path coordinates.

    Args:
        path: Path to the edges GeoPackage file.

    Returns:
        GeoDataFrame in EPSG:4326 with a 'path' column of coordinate lists.
    """
    logging.info(f"Loading edges from {path}...")
    gdf = gpd.read_file(path)

    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf["path"] = gdf.geometry.apply(lambda g: list(g.coords))
    logging.info(f"Loaded {len(gdf)} edges")
    return gdf


def load_nodes(path: Path) -> gpd.GeoDataFrame:
    """Load nodes GeoPackage and extract lon/lat coordinates.

    Args:
        path: Path to the nodes GeoPackage file.

    Returns:
        GeoDataFrame with lon and lat columns added.
    """
    logging.info(f"Loading nodes from {path}...")
    gdf = gpd.read_file(path)

    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    logging.info(f"Loaded {len(gdf)} nodes")
    return gdf


def _normalize_values(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize values to [0, 1] range, clipping to bounds.

    Args:
        values: Array of values to normalize.
        vmin: Minimum bound for normalization.
        vmax: Maximum bound for normalization.

    Returns:
        Array of normalized values in [0, 1].
    """
    if vmax == vmin:
        return np.zeros_like(values, dtype=float)
    clipped = np.clip(values, vmin, vmax)
    return (clipped - vmin) / (vmax - vmin)


def _value_to_rgb(normalized: np.ndarray) -> list[list[int]]:
    """Convert normalized [0,1] values to Plasma colormap RGB colors.

    Args:
        normalized: Array of values in [0, 1].

    Returns:
        List of [R, G, B, A] color arrays.
    """
    cmap = colormaps["plasma"]
    colors = []
    for val in normalized:
        r, g, b, _ = cmap(val)
        colors.append([int(r * 255), int(g * 255), int(b * 255), 180])
    return colors


def _get_normalization_bounds(
    gdf: gpd.GeoDataFrame, column: str
) -> tuple[float, float]:
    """Get normalization bounds for a variable using the appropriate strategy.

    Args:
        gdf: GeoDataFrame with the column.
        column: Column name.

    Returns:
        Tuple of (vmin, vmax).
    """
    values = gdf[column]

    if column == "k_i":
        return 1.0, 10.0
    elif column == "c_i":
        return 0.0, 1.0
    elif column in ("b_i", "avg_l_i"):
        return float(values.min()), float(np.percentile(values.dropna(), 99))
    elif column in ("v_i", "v_ij"):
        return 0.0, float(values.max()) if values.max() > 0 else 1.0
    elif column == "e_ij":
        return float(values.min()), float(np.percentile(values.dropna(), 99))
    else:
        return float(values.min()), float(values.max())


def _plasma_css_gradient(n_stops: int = 10) -> str:
    """Generate a CSS linear-gradient string sampling the Plasma colormap.

    Args:
        n_stops: Number of color stops to sample.

    Returns:
        CSS linear-gradient string (bottom-to-top).
    """
    cmap = colormaps["plasma"]
    stops = []
    for i in range(n_stops):
        t = i / (n_stops - 1)
        r, g, b, _ = cmap(t)
        pct = int(t * 100)
        stops.append(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)}) {pct}%")
    return f"linear-gradient(to top, {', '.join(stops)})"


def _build_overlay_html(label: str, vmin: float, vmax: float) -> str:
    """Build HTML/CSS overlay string with legend and north arrow.

    Args:
        label: Human-readable variable name for the legend title.
        vmin: Minimum value of the color scale.
        vmax: Maximum value of the color scale.

    Returns:
        HTML string to inject before </body>.
    """
    gradient = _plasma_css_gradient()

    vmin_str = f"{vmin:.4g}"
    vmax_str = f"{vmax:.4g}"

    return f"""
<div id="map-legend" style="
    position:absolute; bottom:30px; right:15px; z-index:999;
    background:rgba(0,0,0,0.7); padding:10px 12px; border-radius:6px;
    color:white; font-family:Arial,sans-serif; font-size:12px;
    pointer-events:none;">
  <div style="margin-bottom:6px; font-weight:bold; text-align:center;">{label}</div>
  <div style="display:flex; align-items:stretch; gap:6px;">
    <div style="
        width:18px; height:140px;
        background:{gradient};
        border:1px solid rgba(255,255,255,0.3); border-radius:2px;">
    </div>
    <div style="display:flex; flex-direction:column; justify-content:space-between; font-size:11px;">
      <span>{vmax_str}</span>
      <span>{vmin_str}</span>
    </div>
  </div>
</div>

<div id="north-arrow" style="
    position:absolute; top:15px; right:15px; z-index:999;
    background:rgba(0,0,0,0.7); padding:8px; border-radius:6px;
    color:white; font-family:Arial,sans-serif; text-align:center;
    pointer-events:none;">
  <svg width="30" height="40" viewBox="0 0 30 40">
    <polygon points="15,2 8,22 15,18 22,22" fill="white" stroke="white" stroke-width="0.5"/>
    <polygon points="15,18 8,22 15,38 22,22" fill="rgba(255,255,255,0.3)" stroke="white" stroke-width="0.5"/>
    <text x="15" y="14" text-anchor="middle" fill="black" font-size="10" font-weight="bold">N</text>
  </svg>
</div>
"""


def create_map(
    gdf: gpd.GeoDataFrame, column: str, label: str
) -> tuple[pdk.Deck, float, float]:
    """Create a pydeck map for a given node parameter.

    Args:
        gdf: GeoDataFrame with node data including lon, lat, and the column.
        column: Column name to visualize.
        label: Human-readable label for the tooltip.

    Returns:
        Tuple of (pydeck Deck object, vmin, vmax).
    """
    logging.info(f"Creating map for {column} ({label})...")

    vmin, vmax = _get_normalization_bounds(gdf, column)
    normalized = _normalize_values(gdf[column].values, vmin, vmax)
    colors = _value_to_rgb(normalized)

    df = gdf[["osmid", "lon", "lat", column]].copy()
    df = df.rename(columns={column: "value"})
    df["value"] = df["value"].round(4)
    df["color"] = colors

    view_state = pdk.ViewState(
        latitude=gdf["lat"].mean(),
        longitude=gdf["lon"].mean(),
        zoom=11,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=30,
        radius_min_pixels=1,
        radius_max_pixels=5,
        pickable=True,
    )

    tooltip = {
        "text": f"OSMID: {{osmid}}\n{label}: {{value}}",
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=pdk.map_styles.DARK,
        tooltip=tooltip,
    )

    return deck, vmin, vmax


def create_edge_map(
    gdf: gpd.GeoDataFrame, column: str, label: str
) -> tuple[pdk.Deck, float, float]:
    """Create a pydeck map for a given edge parameter.

    Args:
        gdf: GeoDataFrame with edge data including path coordinates.
        column: Column name to visualize.
        label: Human-readable label for the tooltip.

    Returns:
        Tuple of (pydeck Deck object, vmin, vmax).
    """
    logging.info(f"Creating edge map for {column} ({label})...")

    vmin, vmax = _get_normalization_bounds(gdf, column)
    normalized = _normalize_values(gdf[column].values, vmin, vmax)
    colors = _value_to_rgb(normalized)

    df = gdf[["osmid", "name", "lanes", "path", column]].copy()
    df["name"] = df["name"].fillna("N/A")
    df["lanes"] = df["lanes"].fillna("N/A")
    df = df.rename(columns={column: "value"})
    df["value"] = df["value"].round(4)
    df["color"] = colors

    bounds = gdf.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=11,
    )

    layer = pdk.Layer(
        "PathLayer",
        data=df,
        get_path="path",
        get_color="color",
        width_min_pixels=1,
        width_max_pixels=3,
        pickable=True,
    )

    tooltip = {
        "text": (
            f"OSMID: {{osmid}}\nName: {{name}}\nLanes: {{lanes}}\n{label}: {{value}}"
        ),
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=pdk.map_styles.DARK,
        tooltip=tooltip,
    )

    return deck, vmin, vmax


def save_map(
    deck: pdk.Deck,
    output_path: Path,
    label: str,
    vmin: float,
    vmax: float,
) -> None:
    """Save a pydeck Deck to an HTML file with legend and north arrow.

    Args:
        deck: pydeck Deck object.
        output_path: Path to save the HTML file.
        label: Variable label for the legend.
        vmin: Minimum value of the color scale.
        vmax: Maximum value of the color scale.
    """
    html_str = deck.to_html(as_string=True)
    overlay_html = _build_overlay_html(label, vmin, vmax)
    html_str = html_str.replace("</body>", overlay_html + "</body>")
    Path(output_path).write_text(html_str)
    logging.info(f"Map saved to {output_path}")


def main():
    """Generate interactive pydeck maps for node and edge parameters."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    nodes_path = Path("data/output/nodes.gpkg")
    edges_path = Path("data/output/edges.gpkg")
    maps_dir = Path("maps")
    maps_dir.mkdir(exist_ok=True)

    nodes_gdf = load_nodes(nodes_path)
    edges_gdf = load_edges(edges_path)

    node_variables = [
        ("k_i", "Degree"),
        ("c_i", "Clustering Coefficient"),
        ("b_i", "Betweenness Centrality"),
        ("avg_l_i", "Avg Edge Length"),
        ("v_i", "Vulnerability"),
    ]

    for column, label in node_variables:
        if column not in nodes_gdf.columns:
            logging.warning(f"Column {column} not found in nodes, skipping")
            continue

        plot_gdf = (
            nodes_gdf[nodes_gdf[column] > 0].copy() if column == "v_i" else nodes_gdf
        )
        deck, vmin, vmax = create_map(plot_gdf, column, label)
        output_path = maps_dir / f"map_{column}.html"
        save_map(deck, output_path, label, vmin, vmax)

    edge_variables = [
        ("e_ij", "Edge Betweenness"),
        ("v_ij", "Edge Vulnerability"),
    ]

    for column, label in edge_variables:
        if column not in edges_gdf.columns:
            logging.warning(f"Column {column} not found in edges, skipping")
            continue

        plot_gdf = (
            edges_gdf[edges_gdf[column] > 0].copy() if column == "v_ij" else edges_gdf
        )
        deck, vmin, vmax = create_edge_map(plot_gdf, column, label)
        output_path = maps_dir / f"map_{column}.html"
        save_map(deck, output_path, label, vmin, vmax)

    logging.info("All maps generated successfully")


if __name__ == "__main__":
    main()
