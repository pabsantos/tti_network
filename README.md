# TTI Network Analysis

Geospatial network analysis project for the Tamanduateí basin (TTI). This project loads spatial data, filters intersecting zones, downloads road network data from OpenStreetMap, and calculates various network parameters including vulnerability analysis near hydrological stations.

## Features

- Load and filter origin-destination (OD) zones that intersect with the Tamanduateí basin
- Download road network data from OpenStreetMap using OSMnx
- Calculate network parameters:
  - Node parameters: degree, clustering coefficient, betweenness centrality, average edge length, vulnerability
  - Edge parameters: topological/euclidean/manhattan distances, physical length, edge betweenness centrality, vulnerability
  - Global parameters: network size, averages, maximums, diameter, shortest paths, theoretical random graph comparisons
- **Vulnerability analysis** near hydrological stations using BFS-based global efficiency
- **High-performance computing** using NetworKit (C++ backend) for graph algorithms
- **Parallel processing** using joblib for CPU-intensive operations
- Memory-efficient BFS-based algorithms for large graphs
- Automatic logging to `log/` directory with timestamps
- Export results in multiple formats (GeoPackage, GraphML, text)

## Requirements

- Python 3.11+
- uv package manager
- Docker (optional, for containerized execution)

## Installation

### Local Development

Install dependencies using uv:

```bash
uv sync
```

### Docker

Build the Docker image:

```bash
docker build -t tti-network .
```

Or use docker-compose:

```bash
docker-compose build
```

## Usage

### Local Execution

Run the script directly:

```bash
uv run python main.py
```

### Docker Execution

Using docker-compose (recommended):

```bash
docker-compose run tti-network
```

Using docker directly:

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/cache:/app/cache tti-network
```

## Configuration

### Test Mode

Edit `main.py` to configure test mode:

```python
TEST_RUN = True  # Run test mode with limited districts
TEST_DISTRICTS = [80, 67]  # Districts to include in test
```

When `TEST_RUN = True`, results are saved to `data/test/`
When `TEST_RUN = False`, results are saved to `data/output/`

## Input Data

Place your input files in:
- `data/raw/tti_shape/Microbacias_Tamanduatei.shp` - Basin boundaries
- `data/raw/od_zones/Zonas_2023.shp` - Origin-destination zones
- `data/raw/stations.geojson` - Hydrological stations (with `posto` field for station ID)

## Output Files

The script generates the following outputs:

- `od_zones_filtered.gpkg` - Filtered OD zones (GeoPackage)
- `road_network.graphml` - Road network graph with all parameters
- `nodes.gpkg` - Nodes with calculated parameters (GeoPackage)
- `edges.gpkg` - Edges with calculated parameters (GeoPackage)
- `results.txt` - Summary of global and average network parameters

## Docker Volumes

The docker-compose configuration mounts the following volumes:

- `./data/raw:/app/data/raw:ro` - Input data (read-only)
- `./data/output:/app/data/output` - Production results
- `./data/test:/app/data/test` - Test results
- `./cache:/app/cache` - OSMnx download cache

## Network Parameters

### Node Parameters (Local)
- `k_i`: Degree of node i
- `c_i`: Clustering coefficient of node i
- `b_i`: Betweenness centrality of node i
- `avg_l_i`: Average length of edges connected to node i
- `v_i`: Node vulnerability (efficiency drop when node is isolated)

### Edge Parameters
- `l_topo`: Topological distance (number of edges in shortest path)
- `l_eucl`: Euclidean distance (geodesic straight-line)
- `l_manh`: Manhattan distance
- `length`: Physical road length in meters
- `e_ij`: Edge betweenness centrality
- `v_ij`: Edge vulnerability (efficiency drop when edge is removed)

### Global Parameters
- `N`: Number of nodes
- `L`: Number of edges
- `<k>`: Average degree
- `<c>`: Average clustering coefficient
- `<l_topo>`, `<l_eucl>`, `<l_manh>`, `<length>`: Average distances
- `D`: Network diameter
- Average shortest path lengths (topological, physical, euclidean, manhattan)
- Theoretical random graph G(N,p) parameters: `p`, `k*`, `c*`, `l*`

## Performance

All graph computations use **NetworKit** (C++ backend), providing significant speedup over pure NetworkX.

### Optimized Operations

| Operation | Backend |
|-----------|---------|
| Node betweenness centrality | `nk.centrality.Betweenness` |
| Edge betweenness centrality | `nk.centrality.Betweenness(computeEdgeCentrality=True)` |
| Clustering coefficient | `nk.centrality.LocalClusteringCoefficient` |
| Diameter | `nk.distance.Diameter` |
| Global efficiency | BFS-based computation (`nk.distance.BFS`) |
| Average shortest path | Parallel BFS across all nodes |

### Memory-Efficient Algorithms

- **BFS-based efficiency**: Uses O(N) memory per source instead of O(N^2) for full APSP matrix
- **Incremental vulnerability**: Computes efficiency change for each edge/node removal independently

### Parallel Processing

- **Vulnerability computation**: Uses `joblib` to parallelize across nodes/edges near hydrological stations
- **Average shortest path**: Parallel Dijkstra for weighted path calculations
- **BFS distances**: Parallel BFS from all source nodes for topological distances
