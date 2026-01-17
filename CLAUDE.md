# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a geospatial network analysis project focused on the Tamanduateí basin (TTI). The project loads spatial data (basin boundaries and origin-destination zones), filters intersecting zones, and downloads/analyzes road network data from OpenStreetMap.

## Development Setup

This project uses `uv` as the Python package manager. Python 3.13+ is required.

**Install dependencies:**
```bash
uv sync
```

**Run the main script:**
```bash
uv run python main.py
```

## Project Architecture

### Data Flow

The main pipeline in `main.py` follows this sequence:

1. **Load spatial data**: Basin shapefile (`Microbacias_Tamanduatei.shp`) and OD zones shapefile (`Zonas_2023.shp`)
2. **CRS alignment**: Ensures both GeoDataFrames use the same coordinate reference system
3. **Spatial filtering**: Creates a union of basin geometries and filters OD zones that intersect with the basin
4. **Network download**: Downloads road network from OSM using the filtered zones as boundary (converted to CRS 4326, validated, and buffered)
5. **Network analysis**: Returns the filtered zones and road network graph (NetworkX MultiDiGraph)

### Key Functions

- `load_tti_basin()` / `load_od_zones()`: Load shapefiles into GeoDataFrames
- `ensure_same_crs()`: Handle CRS mismatches by reprojecting or setting CRS
- `create_basin_union()`: Create single geometry from multiple basin polygons
- `filter_intersecting_zones()`: Spatial filter using intersection test
- `load_graph_from_zones()`: Download OSM road network using OSMnx, requires CRS 4326 for polygon input
- `setup_logging()`: Configure logging with timestamp format

### Directory Structure

- `data/raw/tti_shape/`: Tamanduateí basin shapefiles
- `data/raw/od_zones/`: Origin-destination zone shapefiles
- `data/output/`: Output directory for processed data
- `cache/`: OSMnx automatically caches downloaded network data here

### Key Dependencies

- **geopandas**: Spatial data operations and shapefile I/O
- **osmnx**: Download and analyze OpenStreetMap road networks
- **networkx**: Graph data structure and analysis
- **pandas**: Tabular data operations
- **matplotlib**: Visualization (currently commented out in main)

## Known Patterns

- The project uses structured logging throughout with INFO level messages
- Geometry operations include validation (`.make_valid()`) and buffering to handle edge cases
- OSMnx downloads are cached automatically to avoid repeated API calls
- Graph is currently set to `network_type="drive"` for road networks

## Other instructions

- When running a python script, use 'uv run'
- Don't use excessive comments on code. 
- Always document methods using docstrings


