"""Extract road position, direction, and shape features from LineStrings."""
import geopandas as gpd
import numpy as np
import pandas as pd

def extract_geometry_features(roads: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute geographic features for directed road segments.

    Parameters
    ----------
    roads : gpd.GeoDataFrame
        Road LineStrings in PyG node order, with a defined CRS.

    Returns
    -------
    pd.DataFrame
        Midpoints (lon/lat), relative metric positions, bearing sine/cosine,
        and straightness, with the same index as the input roads.

    Raises
    ------
    ValueError
        If the CRS is missing or the geometries are not valid LineStrings.
    """
    if roads.crs is None:
        raise ValueError("Road geometries must have a defined CRS.")
    if roads.geometry.isna().any() or roads.geometry.is_empty.any():
        raise ValueError("Road geometries cannot be null or empty.")
    if not roads.geometry.geom_type.eq("LineString").all():
        raise ValueError("Road geometries must all be LineStrings.")

    # Calculate lengths, midpoint positions, and directions in meters,
    # not in the degree units of geographic coordinate systems.
    metric_crs = roads.estimate_utm_crs()
    if metric_crs is None:
        raise ValueError("Cannot determine a suitable metric UTM CRS.")

    projected = roads.to_crs(metric_crs)
    geometry = projected.geometry

    midpoint = geometry.interpolate(0.5, normalized=True)
    start = geometry.interpolate(0.0, normalized=True)
    end = geometry.interpolate(1.0, normalized=True)

    dx = end.x.to_numpy() - start.x.to_numpy()
    dy = end.y.to_numpy() - start.y.to_numpy()
    chord = np.hypot(dx, dy)
    path_length = geometry.length.to_numpy()

    # Compass bearing: north = (sin=0, cos=1), east = (sin=1, cos=0).
    # Zero-displacement roads receive (0, 0) rather than NaN.
    bearing_sin = np.divide(dx, chord, out=np.zeros_like(dx), where=chord > 0)
    bearing_cos = np.divide(dy, chord, out=np.zeros_like(dy), where=chord > 0)

    # Endpoint distance / path length: 1 for a straight line, lower for bends.
    straightness = np.divide(
        chord, path_length, out=np.zeros_like(chord), where=path_length > 0
    ).clip(0.0, 1.0)

    # Store GPS coordinates independently from optional model-input coordinates.
    geographic_midpoint = midpoint.to_crs("EPSG:4326")
    min_x, min_y, max_x, max_y = projected.total_bounds
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    return pd.DataFrame(
        {
            # Geographic coordinates of each road segment's midpoint (on the road).
            # Coordinates are converted back to EPSG:4326 (degrees) for GPS compatibility.
            "mid_lon": geographic_midpoint.x.to_numpy(),
            "mid_lat": geographic_midpoint.y.to_numpy(),
            
            # Road midpoint's position relative to the network's center, in kilometers.
            # Subtracting the center makes coordinates city-relative instead of absolute.
            # Dividing by 1000 converts projected CRS units (meters) to kilometers.
            "relative_x_km": (midpoint.x.to_numpy() - center_x) / 1000.0,
            "relative_y_km": (midpoint.y.to_numpy() - center_y) / 1000.0,

            # Encode road direction using sine and cosine of its bearing.
            # This avoids the 0°/360° discontinuity and preserves directional information.
            "bearing_sin": bearing_sin,
            "bearing_cos": bearing_cos,
            "straightness": straightness,
        },
        index=roads.index,
    )
