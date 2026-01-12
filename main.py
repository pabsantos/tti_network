import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd


def setup_logging():
    """Configure logging with INFO level and timestamp format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_tti_basin(path: str) -> gpd.GeoDataFrame:
    """Load Tamanduateí basin shapefile.

    Args:
        path: Path to the basin shapefile.

    Returns:
        GeoDataFrame containing basin geometries.
    """
    logging.info("Loading Tamanduateí basin shapefile...")
    tti_basin = gpd.read_file(path)
    logging.info(f"Tamanduateí basin loaded: {len(tti_basin)} geometry(ies)")
    return tti_basin


def load_od_zones(path: str) -> gpd.GeoDataFrame:
    """Load origin-destination zones shapefile.

    Args:
        path: Path to the OD zones shapefile.

    Returns:
        GeoDataFrame containing OD zone geometries.
    """
    logging.info("Loading OD zones shapefile...")
    od_zones = gpd.read_file(path)
    logging.info(f"OD zones loaded: {len(od_zones)} zone(s)")
    return od_zones


def ensure_same_crs(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Ensure both GeoDataFrames use the same coordinate reference system.

    Args:
        gdf1: First GeoDataFrame.
        gdf2: Second GeoDataFrame.

    Returns:
        Tuple of both GeoDataFrames with aligned CRS.
    """
    crs1 = gdf1.crs
    crs2 = gdf2.crs

    if crs1 != crs2:
        logging.warning(f"CRS mismatch detected: {crs1} vs {crs2}")
        if crs1 is None:
            logging.warning("First GeoDataFrame has no CRS, using second CRS")
            gdf1 = gdf1.set_crs(crs2, allow_override=True)
        elif crs2 is None:
            logging.warning("Second GeoDataFrame has no CRS, using first CRS")
            gdf2 = gdf2.set_crs(crs1, allow_override=True)
        else:
            logging.warning(
                f"Reprojecting second GeoDataFrame to match first CRS: {crs1}"
            )
            gdf2 = gdf2.to_crs(crs1)
        logging.info(f"Both GeoDataFrames now have CRS: {gdf1.crs}")
    else:
        logging.info(f"Both GeoDataFrames already have the same CRS: {crs1}")

    return gdf1, gdf2


def create_basin_union(tti_basin: gpd.GeoDataFrame):
    """Create a single unified geometry from multiple basin polygons.

    Args:
        tti_basin: GeoDataFrame containing basin geometries.

    Returns:
        Unified geometry representing the entire basin.
    """
    logging.info("Creating union of basin geometries...")
    tti_basin_union = tti_basin.geometry.union_all()
    logging.info("Basin union created successfully")
    return tti_basin_union


def filter_intersecting_zones(
    od_zones: gpd.GeoDataFrame, basin_union
) -> gpd.GeoDataFrame:
    """Filter OD zones that spatially intersect with the basin.

    Args:
        od_zones: GeoDataFrame of origin-destination zones.
        basin_union: Unified basin geometry to test intersection against.

    Returns:
        Filtered GeoDataFrame containing only intersecting zones.
    """
    logging.info("Filtering OD zones that intersect with the basin...")
    od_zones_filtered = od_zones[od_zones.geometry.intersects(basin_union)]
    logging.info(
        f"Filtering completed: {len(od_zones_filtered)} out of {len(od_zones)} zone(s) intersect with the basin"
    )
    return od_zones_filtered


def load_graph_from_zones(od_zones_filtered: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    """Download road network from OpenStreetMap using filtered zones as boundary.

    Args:
        od_zones_filtered: GeoDataFrame of filtered OD zones.

    Returns:
        NetworkX MultiDiGraph representing the road network.
    """
    logging.info("Creating network graph from polygon...")
    graph_area = (
        od_zones_filtered.to_crs(4326).geometry.make_valid().union_all().buffer(0)
    )
    logging.info("Union of filtered zones created successfully")

    logging.info("Downloading road network graph from OSM...")
    graph = ox.graph_from_polygon(graph_area, network_type="drive")
    logging.info(f"Graph loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    return graph


def plot_graph(graph: nx.MultiDiGraph):
    """Visualize the road network graph.

    Args:
        graph: NetworkX MultiDiGraph to plot.
    """
    logging.info("Plotting graph...")
    ox.plot_graph(graph)
    logging.info("Graph plotted successfully")


def main():
    """Execute the main pipeline: load spatial data, filter intersecting zones, and download road network.

    Returns:
        Tuple containing filtered OD zones and the road network graph.
    """
    setup_logging()

    # Test run configuration
    TEST_RUN = True
    TEST_DISTRICTS = [80, 67]

    PATH_TTI_SHAPE = "data/raw/tti_shape/Microbacias_Tamanduatei.shp"
    PATH_OD_ZONES = "data/raw/od_zones/Zonas_2023.shp"

    tti_basin = load_tti_basin(PATH_TTI_SHAPE)
    od_zones = load_od_zones(PATH_OD_ZONES)

    if TEST_RUN:
        logging.info(f"Test run mode: filtering to districts {TEST_DISTRICTS}")
        od_zones = od_zones[od_zones["NumDistrit"].isin(TEST_DISTRICTS)]
        logging.info(f"Test run: {len(od_zones)} zone(s) after district filter")

    tti_basin, od_zones = ensure_same_crs(tti_basin, od_zones)

    basin_union = create_basin_union(tti_basin)
    od_zones_filtered = filter_intersecting_zones(od_zones, basin_union)
    graph = load_graph_from_zones(od_zones_filtered)

    # plot_graph(graph)

    output_dir = Path("data/test") if TEST_RUN else Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving results to {output_dir}/")

    zones_output = output_dir / "od_zones_filtered.shp"
    od_zones_filtered.to_file(zones_output)
    logging.info(f"Filtered OD zones saved to {zones_output}")

    graph_output = output_dir / "road_network.graphml"
    ox.save_graphml(graph, graph_output)
    logging.info(f"Road network graph saved to {graph_output}")

    return od_zones_filtered, graph


if __name__ == "__main__":
    main()
