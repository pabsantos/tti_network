import logging
import os
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import networkit as nk
import networkx as nx
import osmnx as ox
import pandas as pd
import pyproj
from joblib import Parallel, delayed

GPU_AVAILABLE = False


def setup_gpu():
    """Configure GPU acceleration using nx-cugraph if available."""
    global GPU_AVAILABLE

    try:
        import nx_cugraph as nxcg
        os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"
        os.environ["NETWORKX_BACKEND_PRIORITY"] = "cugraph"
        GPU_AVAILABLE = True
        logging.info(f"GPU acceleration enabled via nx-cugraph (version: {nxcg.__version__})")
        return True
    except ImportError:
        try:
            import networkx as nx
            if "cugraph" in nx.config.backends:
                os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"
                os.environ["NETWORKX_BACKEND_PRIORITY"] = "cugraph"
                GPU_AVAILABLE = True
                logging.info("GPU acceleration enabled via cugraph backend")
                return True
        except Exception:
            pass
        logging.info("nx-cugraph not available, using CPU (NetworKit/NetworkX)")
        return False


def setup_logging():
    """Configure logging with INFO level, console output, and file output."""
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"tti_network_{timestamp}.log"

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Log file: {log_file.absolute()}")


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
    """Calculate global efficiency of the network using NetworkX.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        Global efficiency value.
    """
    undirected = graph.to_undirected()
    return nx.global_efficiency(undirected)


def _nx_to_nk_graph(graph: nx.MultiDiGraph) -> tuple[nk.Graph, dict, dict]:
    """Convert NetworkX MultiDiGraph to NetworKit Graph.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        Tuple of (NetworKit graph, node_to_idx mapping, idx_to_node mapping).
    """
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    nk_graph = nk.Graph(len(node_list), directed=False, weighted=False)
    for u, v in graph.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        if not nk_graph.hasEdge(u_idx, v_idx):
            nk_graph.addEdge(u_idx, v_idx)

    return nk_graph, node_to_idx, idx_to_node


def _calculate_efficiency_networkit(nk_graph: nk.Graph) -> float:
    """Calculate global efficiency using NetworKit with memory-efficient BFS.

    Args:
        nk_graph: NetworKit Graph.

    Returns:
        Global efficiency value.
    """
    n = nk_graph.numberOfNodes()
    if n <= 1:
        return 0.0

    total_efficiency = 0.0
    for u in range(n):
        bfs = nk.distance.BFS(nk_graph, u)
        bfs.run()
        distances = bfs.getDistances()
        for v in range(n):
            if u != v:
                dist = distances[v]
                if dist < 1e308 and dist > 0:
                    total_efficiency += 1.0 / dist

    return total_efficiency / (n * (n - 1))


def _compute_vulnerability_networkit(
    nk_graph: nk.Graph, u_idx: int, v_idx: int, base_efficiency: float
) -> float:
    """Compute vulnerability for a single edge using NetworKit.

    Args:
        nk_graph: NetworKit Graph.
        u_idx: Source node index.
        v_idx: Target node index.
        base_efficiency: Base global efficiency.

    Returns:
        Vulnerability value.
    """
    graph_copy = nk.Graph(nk_graph)
    graph_copy.removeEdge(u_idx, v_idx)

    efficiency_without = _calculate_efficiency_networkit(graph_copy)
    if base_efficiency > 0:
        return (base_efficiency - efficiency_without) / base_efficiency
    return 0.0


def _compute_clustering_networkit(graph: nx.MultiDiGraph) -> dict:
    """Compute local clustering coefficient using NetworKit (faster than NetworkX).

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        Dictionary mapping node IDs to clustering coefficient values.
    """
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    nk_graph = nk.Graph(len(node_list), directed=False)
    for u, v in graph.edges():
        if u == v:
            continue
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        if not nk_graph.hasEdge(u_idx, v_idx):
            nk_graph.addEdge(u_idx, v_idx)

    lcc = nk.centrality.LocalClusteringCoefficient(nk_graph)
    lcc.run()
    scores = lcc.scores()

    return {node: scores[node_to_idx[node]] for node in node_list}


def _compute_betweenness_networkit(graph: nx.MultiDiGraph) -> dict:
    """Compute betweenness centrality using NetworKit (faster than NetworkX).

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        Dictionary mapping node IDs to betweenness centrality values.
    """
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    nk_graph = nk.Graph(len(node_list), directed=True)
    for u, v in graph.edges():
        nk_graph.addEdge(node_to_idx[u], node_to_idx[v])

    bc = nk.centrality.Betweenness(nk_graph, normalized=True)
    bc.run()
    scores = bc.scores()

    return {node: scores[node_to_idx[node]] for node in node_list}


def _compute_betweenness_subset(graph: nx.MultiDiGraph, sources: list) -> dict:
    """Compute betweenness centrality contribution from a subset of source nodes.

    Args:
        graph: NetworkX MultiDiGraph.
        sources: List of source nodes to compute from.

    Returns:
        Dictionary with partial betweenness values for all nodes.
    """
    return nx.betweenness_centrality_subset(
        graph, sources=sources, targets=list(graph.nodes())
    )


def calculate_node_parameters(
    graph: nx.MultiDiGraph, use_networkit: bool = True
) -> pd.DataFrame:
    """Calculate local parameters for each node in the graph.

    Args:
        graph: NetworkX MultiDiGraph.
        use_networkit: Use NetworKit for betweenness (faster). Falls back to parallel NetworkX if False.

    Returns:
        DataFrame with node parameters (k_i, c_i, b_i, avg_l_i).
    """
    logging.info("Calculating node parameters...")

    logging.info("Computing degree for each node...")
    degree = dict(graph.degree())

    if GPU_AVAILABLE:
        logging.info("Computing clustering coefficient using GPU (nx-cugraph)...")
        node_list = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        clean_graph = nx.Graph()
        clean_graph.add_nodes_from(range(len(node_list)))
        for u, v in graph.to_undirected().edges():
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            if u_idx != v_idx and not clean_graph.has_edge(u_idx, v_idx):
                clean_graph.add_edge(u_idx, v_idx)
        clustering_raw = nx.clustering(clean_graph, backend="cugraph")
        clustering = {node_list[idx]: val for idx, val in clustering_raw.items()}
        logging.info("GPU clustering computation completed")
    elif use_networkit:
        logging.info("Computing clustering coefficient using NetworKit...")
        clustering = _compute_clustering_networkit(graph)
        logging.info("NetworKit clustering computation completed")
    else:
        logging.info("Computing clustering coefficient using NetworkX...")
        simple_graph = nx.Graph(graph.to_undirected())
        clustering = nx.clustering(simple_graph)

    if GPU_AVAILABLE:
        logging.info("Computing betweenness centrality using GPU (nx-cugraph)...")
        node_list = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        clean_graph = nx.DiGraph()
        clean_graph.add_nodes_from(range(len(node_list)))
        for u, v in graph.edges():
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            if u_idx != v_idx and not clean_graph.has_edge(u_idx, v_idx):
                clean_graph.add_edge(u_idx, v_idx)
        betweenness_raw = nx.betweenness_centrality(clean_graph, backend="cugraph")
        betweenness = {node_list[idx]: val for idx, val in betweenness_raw.items()}
        logging.info("GPU betweenness computation completed")
    elif use_networkit:
        logging.info("Computing betweenness centrality using NetworKit...")
        betweenness = _compute_betweenness_networkit(graph)
        logging.info("NetworKit betweenness computation completed")
    else:
        logging.info("Computing betweenness centrality using NetworkX (parallel)...")
        nodes = list(graph.nodes())
        n_jobs = os.cpu_count() or 8
        logging.info(f"Using {n_jobs} CPU cores for parallel betweenness computation")

        chunk_size = max(1, len(nodes) // n_jobs)
        node_chunks = [
            nodes[i : i + chunk_size] for i in range(0, len(nodes), chunk_size)
        ]

        partial_betweenness = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_compute_betweenness_subset)(graph, chunk) for chunk in node_chunks
        )

        betweenness = {node: 0.0 for node in nodes}
        for partial in partial_betweenness:
            for node, value in partial.items():
                betweenness[node] += value

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
            "avg_l_i": [avg_edge_length[n] for n in graph.nodes()],
        }
    )

    logging.info(f"Node parameters calculated for {len(node_data)} nodes")
    return node_data


def _compute_edge_betweenness_networkit(graph: nx.MultiDiGraph) -> dict:
    """Compute edge betweenness centrality using NetworKit (faster than NetworkX).

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        Dictionary mapping (u, v) tuples to edge betweenness centrality values.
    """
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    nk_graph = nk.Graph(len(node_list), directed=True)
    edge_map = {}

    for u, v in graph.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        if not nk_graph.hasEdge(u_idx, v_idx):
            nk_graph.addEdge(u_idx, v_idx)
        edge_map[(u_idx, v_idx)] = (u, v)

    nk_graph.indexEdges()
    bc = nk.centrality.Betweenness(
        nk_graph, normalized=True, computeEdgeCentrality=True
    )
    bc.run()

    edge_scores = bc.edgeScores()
    result = {}
    for u_idx, v_idx in edge_map:
        edge_id = nk_graph.edgeId(u_idx, v_idx)
        u, v = idx_to_node[u_idx], idx_to_node[v_idx]
        result[(u, v)] = edge_scores[edge_id]

    return result


def _compute_single_edge_vulnerability(
    graph: nx.MultiDiGraph, u, v, key, base_efficiency: float
) -> float:
    """Compute vulnerability for a single edge by removing it and measuring efficiency drop.

    Args:
        graph: NetworkX MultiDiGraph.
        u: Source node.
        v: Target node.
        key: Edge key.
        base_efficiency: Base global efficiency of the complete graph.

    Returns:
        Vulnerability value for the edge.
    """
    graph_copy = graph.copy()
    graph_copy.remove_edge(u, v, key)

    if len(graph_copy.nodes()) > 0:
        efficiency_without = calculate_global_efficiency(graph_copy)
        return (
            (base_efficiency - efficiency_without) / base_efficiency
            if base_efficiency > 0
            else 0
        )
    return 0


def calculate_edge_parameters(
    graph: nx.MultiDiGraph,
    compute_vulnerability: bool = True,
    use_networkit: bool = True,
) -> pd.DataFrame:
    """Calculate parameters for each edge in the graph.

    Args:
        graph: NetworkX MultiDiGraph.
        compute_vulnerability: Whether to compute edge vulnerability (expensive).
        use_networkit: Use NetworKit for edge betweenness (faster).

    Returns:
        DataFrame with edge parameters (l_topo, l_eucl, l_manh, length, e_ij, v_ij).
    """
    logging.info("Calculating edge parameters...")

    if GPU_AVAILABLE:
        logging.info("Computing edge betweenness centrality using GPU (nx-cugraph)...")
        node_list = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        clean_graph = nx.DiGraph()
        clean_graph.add_nodes_from(range(len(node_list)))
        for u, v in graph.edges():
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            if u_idx != v_idx and not clean_graph.has_edge(u_idx, v_idx):
                clean_graph.add_edge(u_idx, v_idx)
        edge_betweenness_raw = nx.edge_betweenness_centrality(clean_graph, backend="cugraph")
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        edge_betweenness_mapped = {}
        for (u_idx, v_idx), val in edge_betweenness_raw.items():
            edge_betweenness_mapped[(idx_to_node[u_idx], idx_to_node[v_idx])] = val
        edge_betweenness = {}
        for u, v, key in graph.edges(keys=True):
            edge_betweenness[(u, v, key)] = edge_betweenness_mapped.get((u, v), 0)
        logging.info("GPU edge betweenness computation completed")
    elif use_networkit:
        logging.info("Computing edge betweenness centrality using NetworKit...")
        edge_betweenness_raw = _compute_edge_betweenness_networkit(graph)
        edge_betweenness = {}
        for u, v, key in graph.edges(keys=True):
            edge_betweenness[(u, v, key)] = edge_betweenness_raw.get((u, v), 0)
        logging.info("NetworKit edge betweenness computation completed")
    else:
        logging.info("Computing edge betweenness centrality using NetworkX...")
        edge_betweenness = nx.edge_betweenness_centrality(graph)

    edges_list = list(graph.edges(keys=True, data=True))
    n_edges = len(edges_list)

    logging.info("Converting graph to NetworKit format...")
    nk_graph, node_to_idx, idx_to_node = _nx_to_nk_graph(graph)

    if compute_vulnerability:
        logging.info("Computing edge vulnerability using NetworKit (parallel)...")

        logging.info("Computing base global efficiency...")
        base_efficiency = _calculate_efficiency_networkit(nk_graph)
        logging.info(f"Base global efficiency: {base_efficiency:.6f}")

        edge_indices = []
        for u, v, key, _ in edges_list:
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            edge_indices.append((u_idx, v_idx))

        n_jobs = os.cpu_count() or 8
        logging.info(f"Using {n_jobs} CPU cores for parallel vulnerability computation")

        vulnerabilities = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_compute_vulnerability_networkit)(
                nk_graph, u_idx, v_idx, base_efficiency
            )
            for u_idx, v_idx in edge_indices
        )
        logging.info("Parallel vulnerability computation completed")
    else:
        logging.info("Skipping edge vulnerability calculation (disabled)")
        vulnerabilities = [0.0] * n_edges

    logging.info("Computing distance metrics for edges...")
    edges_data = []
    geod = pyproj.Geod(ellps="WGS84")

    for i, ((u, v, key, data), vuln) in enumerate(zip(edges_list, vulnerabilities), 1):
        if i % 500 == 0 or i == n_edges:
            logging.info(f"Processing edge metrics {i}/{n_edges}")

        u_node = graph.nodes[u]
        v_node = graph.nodes[v]
        u_x, u_y = u_node.get("x"), u_node.get("y")
        v_x, v_y = v_node.get("x"), v_node.get("y")

        l_topo = 1

        length = data.get("length", 0)

        if u_x is not None and u_y is not None and v_x is not None and v_y is not None:
            _, _, l_eucl = geod.inv(u_x, u_y, v_x, v_y)
            _, _, dx = geod.inv(u_x, u_y, u_x, v_y)
            _, _, dy = geod.inv(u_x, u_y, v_x, u_y)
            l_manh = abs(dx) + abs(dy)
        else:
            l_eucl = 0
            l_manh = 0

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
                "v_ij": vuln,
            }
        )

    edge_df = pd.DataFrame(edges_data)
    logging.info(f"Edge parameters calculated for {len(edge_df)} edges")
    return edge_df


def calculate_global_parameters(
    graph: nx.MultiDiGraph,
    node_data: pd.DataFrame,
    edge_data: pd.DataFrame,
    use_networkit: bool = True,
) -> dict:
    """Calculate global network parameters.

    Args:
        graph: NetworkX MultiDiGraph.
        node_data: DataFrame with node parameters.
        edge_data: DataFrame with edge parameters.
        use_networkit: Use NetworKit for diameter and avg shortest path (faster).

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
    max_edge_vulnerability = edge_data["v_ij"].max()

    largest_cc = max(nx.weakly_connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_cc)

    if use_networkit:
        logging.info("Computing diameter and avg shortest path using NetworKit...")
        nk_subgraph, node_to_idx, _ = _nx_to_nk_graph(subgraph)

        try:
            diam = nk.distance.Diameter(
                nk_subgraph, algo=nk.distance.DiameterAlgo.EXACT
            )
            diam.run()
            diameter = int(diam.getDiameter()[0])
            logging.info(
                f"Diameter calculated on largest connected component with {len(subgraph.nodes())} nodes"
            )
        except Exception as e:
            logging.warning(f"NetworKit diameter failed: {e}, falling back to NetworkX")
            try:
                diameter = nx.diameter(subgraph.to_undirected())
                logging.info(
                    f"Diameter calculated using NetworkX on {len(subgraph.nodes())} nodes"
                )
            except nx.NetworkXError:
                diameter = None
                logging.warning(
                    "Could not calculate diameter (graph may not be connected)"
                )

        try:
            n = nk_subgraph.numberOfNodes()
            total_dist = 0.0
            count = 0
            for u in range(n):
                bfs = nk.distance.BFS(nk_subgraph, u)
                bfs.run()
                distances = bfs.getDistances()
                for v in range(u + 1, n):
                    dist = distances[v]
                    if dist < 1e308:
                        total_dist += dist
                        count += 1
            avg_shortest_path_topo = total_dist / count if count > 0 else None
            logging.info("Average shortest path length (topological) calculated")
        except Exception as e:
            avg_shortest_path_topo = None
            logging.warning(
                f"Could not calculate average shortest path length (topological): {e}"
            )
    else:
        try:
            diameter = nx.diameter(subgraph.to_undirected())
            logging.info(
                f"Diameter calculated on largest connected component with {len(subgraph.nodes())} nodes"
            )
        except nx.NetworkXError:
            diameter = None
            logging.warning("Could not calculate diameter (graph may not be connected)")

        undirected_subgraph = graph.to_undirected().subgraph(largest_cc)
        try:
            avg_shortest_path_topo = nx.average_shortest_path_length(
                undirected_subgraph
            )
            logging.info("Average shortest path length (topological) calculated")
        except nx.NetworkXError:
            avg_shortest_path_topo = None
            logging.warning(
                "Could not calculate average shortest path length (topological)"
            )

    undirected_subgraph = graph.to_undirected().subgraph(largest_cc)

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
    setup_gpu()

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
