import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import pyproj


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


def calculate_global_efficiency(graph: nx.MultiDiGraph) -> float:
    """Calculate global efficiency of the network.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        Global efficiency value.
    """
    undirected = graph.to_undirected()
    return nx.global_efficiency(undirected)


def calculate_node_parameters(
    graph: nx.MultiDiGraph, compute_vulnerability: bool = True
) -> pd.DataFrame:
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

    logging.info("Computing average edge length for each node...")
    avg_edge_length = {}
    for node in graph.nodes():
        edges = graph.edges(node, data=True)
        lengths = [data.get("length", 0) for _, _, data in edges]
        avg_edge_length[node] = sum(lengths) / len(lengths) if lengths else 0

    if compute_vulnerability:
        logging.info("Computing node vulnerability based on efficiency...")
        base_efficiency = calculate_global_efficiency(graph)
        logging.info(f"Base global efficiency: {base_efficiency:.6f}")

        vulnerability = {}
        nodes_list = list(graph.nodes())
        for i, node in enumerate(nodes_list, 1):
            if i % 100 == 0 or i == len(nodes_list):
                logging.info(f"Computing vulnerability for node {i}/{len(nodes_list)}")

            graph_copy = graph.copy()
            graph_copy.remove_node(node)

            if len(graph_copy.nodes()) > 0:
                efficiency_without = calculate_global_efficiency(graph_copy)
                vulnerability[node] = (
                    (base_efficiency - efficiency_without) / base_efficiency
                    if base_efficiency > 0
                    else 0
                )
            else:
                vulnerability[node] = 0
    else:
        logging.info("Skipping node vulnerability calculation (disabled for test run)")
        vulnerability = {n: 0.0 for n in graph.nodes()}

    node_data = pd.DataFrame(
        {
            "node": list(graph.nodes()),
            "k_i": [degree[n] for n in graph.nodes()],
            "c_i": [clustering[n] for n in graph.nodes()],
            "b_i": [betweenness[n] for n in graph.nodes()],
            "v_i": [vulnerability[n] for n in graph.nodes()],
            "avg_l_i": [avg_edge_length[n] for n in graph.nodes()],
        }
    )

    logging.info(f"Node parameters calculated for {len(node_data)} nodes")
    return node_data


def calculate_edge_parameters(
    graph: nx.MultiDiGraph, compute_vulnerability: bool = True
) -> pd.DataFrame:
    """Calculate parameters for each edge in the graph.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        DataFrame with edge parameters (l_topo, l_eucl, l_manh, length, e_ij, v_ij).
    """
    logging.info("Calculating edge parameters...")

    logging.info("Computing edge betweenness centrality...")
    edge_betweenness = nx.edge_betweenness_centrality(graph)

    logging.info("Computing shortest path lengths for topological distance...")
    # Create undirected version for shortest path calculations
    undirected = graph.to_undirected()

    logging.info("Computing distance metrics for edges...")
    edges_data = []
    edges_list = list(graph.edges(keys=True, data=True))

    if compute_vulnerability:
        logging.info("Computing edge vulnerability based on efficiency...")
        base_efficiency = calculate_global_efficiency(graph)
        logging.info(f"Base global efficiency: {base_efficiency:.6f}")
    else:
        logging.info("Skipping edge vulnerability calculation (disabled for test run)")
        base_efficiency = None

    for i, (u, v, key, data) in enumerate(edges_list, 1):
        if i % 200 == 0 or i == len(edges_list):
            logging.info(f"Processing edge {i}/{len(edges_list)}")

        # Get node coordinates
        u_node = graph.nodes[u]
        v_node = graph.nodes[v]
        u_x, u_y = u_node.get("x"), u_node.get("y")
        v_x, v_y = v_node.get("x"), v_node.get("y")

        # Topological distance (shortest path length in number of edges)
        try:
            l_topo = nx.shortest_path_length(undirected, source=u, target=v)
        except nx.NetworkXNoPath:
            l_topo = float("inf")

        # Physical length (actual road network distance in meters)
        length = data.get("length", 0)

        # Euclidean distance (straight-line) and Manhattan distance
        if u_x is not None and u_y is not None and v_x is not None and v_y is not None:
            # Use WGS84 ellipsoid for accurate distance calculations
            geod = pyproj.Geod(ellps="WGS84")

            # Euclidean distance (geodesic straight-line distance)
            _, _, l_eucl = geod.inv(u_x, u_y, v_x, v_y)

            # Manhattan distance (sum of absolute differences)
            # Convert lat/lon differences to meters
            _, _, dx = geod.inv(u_x, u_y, u_x, v_y)
            _, _, dy = geod.inv(u_x, u_y, v_x, u_y)
            l_manh = abs(dx) + abs(dy)
        else:
            l_eucl = 0
            l_manh = 0

        # Calculate vulnerability for this edge
        if compute_vulnerability:
            graph_copy = graph.copy()
            graph_copy.remove_edge(u, v, key)

            if len(graph_copy.nodes()) > 0:
                efficiency_without = calculate_global_efficiency(graph_copy)
                edge_vulnerability = (
                    (base_efficiency - efficiency_without) / base_efficiency
                    if base_efficiency > 0
                    else 0
                )
            else:
                edge_vulnerability = 0
        else:
            edge_vulnerability = 0.0

        edges_data.append(
            {
                "u": u,
                "v": v,
                "key": key,
                "l_topo": l_topo,
                "l_eucl": l_eucl,
                "l_manh": l_manh,
                "length": length,
                "e_ij": edge_betweenness.get((u, v, key), 0),
                "v_ij": edge_vulnerability,
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
    avg_l_topo = edge_data["l_topo"].replace(float("inf"), -1).mean()
    avg_l_eucl = edge_data["l_eucl"].mean()
    avg_l_manh = edge_data["l_manh"].mean()
    avg_length = edge_data["length"].mean()

    max_degree = node_data["k_i"].max()
    max_clustering = node_data["c_i"].max()
    max_l_topo = int(edge_data["l_topo"].replace(float("inf"), -1).max())
    max_l_eucl = edge_data["l_eucl"].max()
    max_l_manh = edge_data["l_manh"].max()
    max_length = edge_data["length"].max()
    max_node_vulnerability = node_data["v_i"].max()
    max_edge_vulnerability = edge_data["v_ij"].max()

    largest_cc = max(nx.weakly_connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_cc)

    try:
        diameter = nx.diameter(subgraph.to_undirected())
        logging.info(
            f"Diameter calculated on largest connected component with {len(subgraph.nodes())} nodes"
        )
    except nx.NetworkXError:
        diameter = None
        logging.warning("Could not calculate diameter (graph may not be connected)")

    # Calculate average shortest path length using different distance metrics
    # Create undirected graph for path calculations
    undirected_subgraph = graph.to_undirected().subgraph(largest_cc)

    try:
        # Topological (number of edges)
        avg_shortest_path_topo = nx.average_shortest_path_length(undirected_subgraph)
        logging.info("Average shortest path length (topological) calculated")
    except nx.NetworkXError:
        avg_shortest_path_topo = None
        logging.warning(
            "Could not calculate average shortest path length (topological)"
        )

    try:
        # Using physical length as weight
        avg_shortest_path_length = nx.average_shortest_path_length(
            undirected_subgraph, weight="length"
        )
        logging.info("Average shortest path length (physical) calculated")
    except nx.NetworkXError:
        avg_shortest_path_length = None
        logging.warning("Could not calculate average shortest path length (physical)")

    try:
        # Using Euclidean distance as weight
        avg_shortest_path_eucl = nx.average_shortest_path_length(
            undirected_subgraph, weight="l_eucl"
        )
        logging.info("Average shortest path length (Euclidean) calculated")
    except nx.NetworkXError:
        avg_shortest_path_eucl = None
        logging.warning("Could not calculate average shortest path length (Euclidean)")

    try:
        # Using Manhattan distance as weight
        avg_shortest_path_manh = nx.average_shortest_path_length(
            undirected_subgraph, weight="l_manh"
        )
        logging.info("Average shortest path length (Manhattan) calculated")
    except nx.NetworkXError:
        avg_shortest_path_manh = None
        logging.warning("Could not calculate average shortest path length (Manhattan)")

    global_params = {
        "N": N,
        "L": L,
        "avg_k": avg_degree,
        "avg_c": avg_clustering,
        "avg_l_topo": avg_l_topo,
        "avg_l_eucl": avg_l_eucl,
        "avg_l_manh": avg_l_manh,
        "avg_length": avg_length,
        "max_k": max_degree,
        "max_c": max_clustering,
        "max_l_topo": max_l_topo,
        "max_l_eucl": max_l_eucl,
        "max_l_manh": max_l_manh,
        "max_length": max_length,
        "max_v_node": max_node_vulnerability,
        "max_v_edge": max_edge_vulnerability,
        "diameter_D": diameter,
        "avg_shortest_path_topo": avg_shortest_path_topo,
        "avg_shortest_path_length": avg_shortest_path_length,
        "avg_shortest_path_eucl": avg_shortest_path_eucl,
        "avg_shortest_path_manh": avg_shortest_path_manh,
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
            l_topo_val = row["l_topo"]
            graph.edges[u, v, key]["l_topo"] = (
                int(l_topo_val) if l_topo_val != float("inf") else -1
            )
            graph.edges[u, v, key]["l_eucl"] = float(row["l_eucl"])
            graph.edges[u, v, key]["l_manh"] = float(row["l_manh"])
            graph.edges[u, v, key]["length"] = float(row["length"])
            graph.edges[u, v, key]["e_ij"] = float(row["e_ij"])
            graph.edges[u, v, key]["v_ij"] = float(row["v_ij"])

    logging.info("Parameters added to graph")
    return graph


def graph_to_spatial_objects(
    graph: nx.MultiDiGraph,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Convert graph nodes and edges to spatial GeoDataFrames.

    Args:
        graph: NetworkX MultiDiGraph with spatial data.

    Returns:
        Tuple of (nodes GeoDataFrame, edges GeoDataFrame).
    """
    logging.info("Converting graph to spatial objects...")

    nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False)

    logging.info(
        f"Created spatial objects: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges"
    )
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
        f.write(
            "Average clustering coefficient (<c>): {:.4f}\n".format(
                global_params["avg_c"]
            )
        )
        f.write(
            "Average topological distance (<l_topo>): {:.4f} edges\n".format(
                global_params["avg_l_topo"]
            )
        )
        f.write(
            "Average Euclidean distance (<l_eucl>): {:.4f} m\n".format(
                global_params["avg_l_eucl"]
            )
        )
        f.write(
            "Average Manhattan distance (<l_manh>): {:.4f} m\n".format(
                global_params["avg_l_manh"]
            )
        )
        f.write(
            "Average physical length (<length>): {:.4f} m\n".format(
                global_params["avg_length"]
            )
        )
        f.write("\n")

        f.write("Average Shortest Path Lengths:\n")
        f.write("-" * 50 + "\n")
        if global_params["avg_shortest_path_topo"] is not None:
            f.write(
                "Topological (edges): {:.4f}\n".format(
                    global_params["avg_shortest_path_topo"]
                )
            )
        else:
            f.write("Topological (edges): N/A (graph not fully connected)\n")

        if global_params["avg_shortest_path_length"] is not None:
            f.write(
                "Physical length (m): {:.4f}\n".format(
                    global_params["avg_shortest_path_length"]
                )
            )
        else:
            f.write("Physical length (m): N/A (graph not fully connected)\n")

        if global_params["avg_shortest_path_eucl"] is not None:
            f.write(
                "Euclidean distance (m): {:.4f}\n".format(
                    global_params["avg_shortest_path_eucl"]
                )
            )
        else:
            f.write("Euclidean distance (m): N/A (graph not fully connected)\n")

        if global_params["avg_shortest_path_manh"] is not None:
            f.write(
                "Manhattan distance (m): {:.4f}\n".format(
                    global_params["avg_shortest_path_manh"]
                )
            )
        else:
            f.write("Manhattan distance (m): N/A (graph not fully connected)\n")
        f.write("\n")

        f.write("Maximum Parameters:\n")
        f.write("-" * 50 + "\n")
        f.write("Maximum degree (k*): {}\n".format(global_params["max_k"]))
        f.write(
            "Maximum clustering coefficient (c*): {:.4f}\n".format(
                global_params["max_c"]
            )
        )
        f.write(
            "Maximum topological distance (l_topo*): {} edges\n".format(
                global_params["max_l_topo"]
            )
        )
        f.write(
            "Maximum Euclidean distance (l_eucl*): {:.4f} m\n".format(
                global_params["max_l_eucl"]
            )
        )
        f.write(
            "Maximum Manhattan distance (l_manh*): {:.4f} m\n".format(
                global_params["max_l_manh"]
            )
        )
        f.write(
            "Maximum physical length (length*): {:.4f} m\n".format(
                global_params["max_length"]
            )
        )
        f.write(
            "Maximum node vulnerability (v*): {:.6f}\n".format(
                global_params["max_v_node"]
            )
        )
        f.write(
            "Maximum edge vulnerability (v_edge*): {:.6f}\n".format(
                global_params["max_v_edge"]
            )
        )
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

    compute_vuln = not TEST_RUN
    node_params = calculate_node_parameters(graph, compute_vulnerability=compute_vuln)
    edge_params = calculate_edge_parameters(graph, compute_vulnerability=compute_vuln)

    # Add parameters to graph BEFORE calculating global parameters
    # so that edge attributes are available for weighted shortest paths
    graph = add_parameters_to_graph(graph, node_params, edge_params)

    global_params = calculate_global_parameters(graph, node_params, edge_params)
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
