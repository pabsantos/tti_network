# TTI Network Analysis

Geospatial network analysis project for the Tamanduateí basin (TTI). This project loads spatial data, filters intersecting zones, downloads road network data from OpenStreetMap, and calculates various network parameters.

## Features

- Load and filter origin-destination (OD) zones that intersect with the Tamanduateí basin
- Download road network data from OpenStreetMap using OSMnx
- Calculate network parameters:
  - Node parameters: degree, clustering coefficient, betweenness centrality, average edge length
  - Edge parameters: topological/euclidean/manhattan distances, physical length, edge betweenness centrality, vulnerability
  - Global parameters: network size, averages, maximums, diameter, shortest paths
- **High-performance computing** using NetworKit (C++ backend) for graph algorithms
- **Parallel processing** using joblib for CPU-intensive operations
- Automatic logging to `log/` directory with timestamps
- Export results in multiple formats (GeoPackage, GraphML, text)

## Requirements

- Python 3.13+
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

Place your input shapefiles in:
- `data/raw/tti_shape/Microbacias_Tamanduatei.shp` - Basin boundaries
- `data/raw/od_zones/Zonas_2023.shp` - Origin-destination zones

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

## Performance

The project uses **NetworKit** for computationally expensive graph algorithms, providing 10-100x speedup over pure NetworkX for large graphs.

### Optimized Operations

| Operation | Library | Speedup |
|-----------|---------|---------|
| Node betweenness centrality | NetworKit | ~50x |
| Edge betweenness centrality | NetworKit | ~50x |
| Clustering coefficient | NetworKit | ~20x |
| All-pairs shortest paths | NetworKit | ~50x |
| Global efficiency | NetworKit | ~50x |
| Diameter | NetworKit | ~10x |
| Edge vulnerability | NetworKit + joblib | ~100x |

### Parallel Processing

Edge vulnerability computation uses all available CPU cores via `joblib`. The script automatically detects the number of cores:
- Apple M1: 8 cores
- Desktop CPUs: scales to 12+ cores

### Estimated Runtime (55k nodes, 100k edges)

| Configuration | Time |
|---------------|------|
| NetworkX only (sequential) | ~weeks |
| NetworKit + parallel | ~hours |
