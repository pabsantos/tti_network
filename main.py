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


def calculate_node_parameters(graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Calculate local parameters for each node in the graph.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        DataFrame with node parameters (k_i, c_i, b_i, v_i, avg_l_i).
    """
    logging.info("Calculating node parameters...")

    logging.info("Computing degree for each node...")
    degree = dict(graph.degree())

    logging.info("Computing clustering coefficient for each node...")
    simple_graph = nx.Graph(graph.to_undirected())
    clustering = nx.clustering(simple_graph)

    logging.info("Computing betweenness centrality for each node...")
    betweenness = nx.betweenness_centrality(graph)

    logging.info("Computing closeness centrality for each node...")
    closeness = nx.closeness_centrality(graph)

    logging.info("Computing average edge length for each node...")
    avg_edge_length = {}
    for node in graph.nodes():
        edges = graph.edges(node, data=True)
        lengths = [data.get("length", 0) for _, _, data in edges]
        avg_edge_length[node] = sum(lengths) / len(lengths) if lengths else 0

    node_data = pd.DataFrame(
        {
            "node": list(graph.nodes()),
            "k_i": [degree[n] for n in graph.nodes()],
            "c_i": [clustering[n] for n in graph.nodes()],
            "b_i": [betweenness[n] for n in graph.nodes()],
            "v_i": [closeness[n] for n in graph.nodes()],
            "avg_l_i": [avg_edge_length[n] for n in graph.nodes()],
        }
    )

    logging.info(f"Node parameters calculated for {len(node_data)} nodes")
    return node_data


def calculate_edge_parameters(graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Calculate parameters for each edge in the graph.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        DataFrame with edge parameters (l_ij, e_ij).
    """
    logging.info("Calculating edge parameters...")

    logging.info("Computing edge betweenness centrality...")
    edge_betweenness = nx.edge_betweenness_centrality(graph)

    edges_data = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        edges_data.append(
            {
                "u": u,
                "v": v,
                "key": key,
                "l_ij": data.get("length", 0),
                "e_ij": edge_betweenness.get((u, v, key), 0),
            }
        )

    edge_df = pd.DataFrame(edges_data)
    logging.info(f"Edge parameters calculated for {len(edge_df)} edges")
    return edge_df


def calculate_global_parameters(
    graph: nx.MultiDiGraph, node_data: pd.DataFrame, edge_data: pd.DataFrame
) -> dict:
    """Calculate global network parameters.

    Args:
        graph: NetworkX MultiDiGraph.
        node_data: DataFrame with node parameters.
        edge_data: DataFrame with edge parameters.

    Returns:
        Dictionary with global parameters.
    """
    logging.info("Calculating global parameters...")

    N = len(graph.nodes())
    L = len(graph.edges())

    avg_degree = node_data["k_i"].mean()
    avg_clustering = node_data["c_i"].mean()
    avg_edge_length = edge_data["l_ij"].mean()

    max_degree = node_data["k_i"].max()
    max_clustering = node_data["c_i"].max()
    max_edge_length = edge_data["l_ij"].max()

    largest_cc = max(nx.weakly_connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_cc)

    try:
        diameter = nx.diameter(subgraph.to_undirected())
        logging.info(f"Diameter calculated on largest connected component with {len(subgraph.nodes())} nodes")
    except nx.NetworkXError:
        diameter = None
        logging.warning("Could not calculate diameter (graph may not be connected)")

    try:
        avg_shortest_path = nx.average_shortest_path_length(subgraph.to_undirected())
        logging.info("Average shortest path length calculated")
    except nx.NetworkXError:
        avg_shortest_path = None
        logging.warning("Could not calculate average shortest path length")

    global_params = {
        "N": N,
        "L": L,
        "avg_k": avg_degree,
        "avg_c": avg_clustering,
        "avg_l": avg_edge_length,
        "max_k": max_degree,
        "max_c": max_clustering,
        "max_l": max_edge_length,
        "diameter_D": diameter,
        "avg_shortest_path": avg_shortest_path,
    }

    logging.info(f"Global parameters calculated: N={N}, L={L}")
    return global_params


def add_parameters_to_graph(
    graph: nx.MultiDiGraph, node_data: pd.DataFrame, edge_data: pd.DataFrame
) -> nx.MultiDiGraph:
    """Add calculated parameters as node and edge attributes to the graph.

    Args:
        graph: NetworkX MultiDiGraph.
        node_data: DataFrame with node parameters.
        edge_data: DataFrame with edge parameters.

    Returns:
        Graph with added attributes.
    """
    logging.info("Adding parameters to graph as attributes...")

    for _, row in node_data.iterrows():
        node = row["node"]
        graph.nodes[node]["k_i"] = int(row["k_i"])
        graph.nodes[node]["c_i"] = float(row["c_i"])
        graph.nodes[node]["b_i"] = float(row["b_i"])
        graph.nodes[node]["v_i"] = float(row["v_i"])
        graph.nodes[node]["avg_l_i"] = float(row["avg_l_i"])

    for _, row in edge_data.iterrows():
        u, v, key = row["u"], row["v"], int(row["key"])
        if graph.has_edge(u, v, key):
            graph.edges[u, v, key]["e_ij"] = float(row["e_ij"])

    logging.info("Parameters added to graph")
    return graph


def graph_to_spatial_objects(graph: nx.MultiDiGraph) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Convert graph nodes and edges to spatial GeoDataFrames.

    Args:
        graph: NetworkX MultiDiGraph with spatial data.

    Returns:
        Tuple of (nodes GeoDataFrame, edges GeoDataFrame).
    """
    logging.info("Converting graph to spatial objects...")

    nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False)

    logging.info(f"Created spatial objects: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges")
    return nodes_gdf, edges_gdf


def save_results_txt(global_params: dict, output_path: Path):
    """Save global and average parameters to a text file.

    Args:
        global_params: Dictionary with global parameters.
        output_path: Path to save the results.txt file.
    """
    logging.info(f"Writing results to {output_path}")

    with open(output_path, "w") as f:
        f.write("Global Network Parameters\n")
        f.write("=" * 50 + "\n\n")

        f.write("Number of nodes (N): {}\n".format(global_params["N"]))
        f.write("Number of edges (L): {}\n\n".format(global_params["L"]))

        f.write("Average Parameters:\n")
        f.write("-" * 50 + "\n")
        f.write("Average degree (<k>): {:.4f}\n".format(global_params["avg_k"]))
        f.write("Average clustering coefficient (<c>): {:.4f}\n".format(global_params["avg_c"]))
        f.write("Average edge length (<l>): {:.4f}\n".format(global_params["avg_l"]))
        if global_params["avg_shortest_path"] is not None:
            f.write("Average shortest path length: {:.4f}\n".format(global_params["avg_shortest_path"]))
        else:
            f.write("Average shortest path length: N/A (graph not fully connected)\n")
        f.write("\n")

        f.write("Maximum Parameters:\n")
        f.write("-" * 50 + "\n")
        f.write("Maximum degree (k*): {}\n".format(global_params["max_k"]))
        f.write("Maximum clustering coefficient (c*): {:.4f}\n".format(global_params["max_c"]))
        f.write("Maximum edge length (l*): {:.4f}\n".format(global_params["max_l"]))
        if global_params["diameter_D"] is not None:
            f.write("Diameter (D): {}\n".format(global_params["diameter_D"]))
        else:
            f.write("Diameter (D): N/A (graph not fully connected)\n")

    logging.info(f"Results saved to {output_path}")


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

    node_params = calculate_node_parameters(graph)
    edge_params = calculate_edge_parameters(graph)
    global_params = calculate_global_parameters(graph, node_params, edge_params)

    graph = add_parameters_to_graph(graph, node_params, edge_params)
    nodes_gdf, edges_gdf = graph_to_spatial_objects(graph)

    output_dir = Path("data/test") if TEST_RUN else Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving results to {output_dir}/")

    zones_output = output_dir / "od_zones_filtered.gpkg"
    od_zones_filtered.to_file(zones_output, driver="GPKG")
    logging.info(f"Filtered OD zones saved to {zones_output}")

    graph_output = output_dir / "road_network.graphml"
    ox.save_graphml(graph, graph_output)
    logging.info(f"Road network graph saved to {graph_output}")

    nodes_spatial_output = output_dir / "nodes.gpkg"
    nodes_gdf.to_file(nodes_spatial_output, driver="GPKG")
    logging.info(f"Nodes spatial object saved to {nodes_spatial_output}")

    edges_spatial_output = output_dir / "edges.gpkg"
    edges_gdf.to_file(edges_spatial_output, driver="GPKG")
    logging.info(f"Edges spatial object saved to {edges_spatial_output}")

    results_txt_output = output_dir / "results.txt"
    save_results_txt(global_params, results_txt_output)

    return od_zones_filtered, graph


if __name__ == "__main__":
    main()
