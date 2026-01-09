import geopandas as gpd
import logging
import networkx as nx
import osmnx as ox
import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_tti_basin(path: str) -> gpd.GeoDataFrame:
    logging.info("Loading Tamanduateí basin shapefile...")
    tti_basin = gpd.read_file(path)
    logging.info(f"Tamanduateí basin loaded: {len(tti_basin)} geometry(ies)")
    return tti_basin


def load_od_zones(path: str) -> gpd.GeoDataFrame:
    logging.info("Loading OD zones shapefile...")
    od_zones = gpd.read_file(path)
    logging.info(f"OD zones loaded: {len(od_zones)} zone(s)")
    return od_zones


def ensure_same_crs(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
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
    logging.info("Creating union of basin geometries...")
    tti_basin_union = tti_basin.geometry.union_all()
    logging.info("Basin union created successfully")
    return tti_basin_union


def filter_intersecting_zones(
    od_zones: gpd.GeoDataFrame, basin_union
) -> gpd.GeoDataFrame:
    logging.info("Filtering OD zones that intersect with the basin...")
    od_zones_filtered = od_zones[od_zones.geometry.intersects(basin_union)]
    logging.info(
        f"Filtering completed: {len(od_zones_filtered)} out of {len(od_zones)} zone(s) intersect with the basin"
    )
    return od_zones_filtered


def load_graph_from_zones(od_zones_filtered: gpd.GeoDataFrame) -> nx.MultiDiGraph:
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
    logging.info("Plotting graph...")
    ox.plot_graph(graph)
    logging.info("Graph plotted successfully")


def main():
    setup_logging()

    PATH_TTI_SHAPE = "data/raw/tti_shape/Microbacias_Tamanduatei.shp"
    PATH_OD_ZONES = "data/raw/od_zones/Zonas_2023.shp"

    tti_basin = load_tti_basin(PATH_TTI_SHAPE)
    od_zones = load_od_zones(PATH_OD_ZONES)

    tti_basin, od_zones = ensure_same_crs(tti_basin, od_zones)

    basin_union = create_basin_union(tti_basin)
    od_zones_filtered = filter_intersecting_zones(od_zones, basin_union)
    graph = load_graph_from_zones(od_zones_filtered)

    # plot_graph(graph)

    return od_zones_filtered, graph


if __name__ == "__main__":
    main()
