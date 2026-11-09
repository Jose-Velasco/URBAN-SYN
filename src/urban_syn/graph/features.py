"""Encode explicitly selected OSM and geographic road-node features."""
from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .geometry import extract_geometry_features
from .osm import (
    osm_category,
    parse_osm_lanes,
    parse_osm_oneway,
    parse_osm_structure,
    parse_osm_width,
)

NUMERIC_FEATURES = frozenset({"log_length", "speed_kph", "lanes", "width"})
CATEGORICAL_FEATURES = frozenset({"highway", "surface", "access", "junction"})
BOOLEAN_FEATURES = frozenset({"oneway", "bridge", "tunnel"})
GEOMETRY_FEATURES = frozenset({
    "mid_lon", "mid_lat", "relative_x_km", "relative_y_km",
    "bearing_sin", "bearing_cos", "straightness",
})


def prepare_feature_table(roads: gpd.GeoDataFrame) -> pd.DataFrame:
    """Clean OSM road tags and append derived geographic attributes.

    Parameters
    ----------
    roads : gpd.GeoDataFrame
        Canonical roads in node order, with geometry and numeric attributes.

    Returns
    -------
    pd.DataFrame
        All supported cleaned attributes, aligned with road rows.

    Raises
    ------
    ValueError
        If essential numeric road attributes are missing or invalid.
    """
    missing = {"length", "speed_kph"} - set(roads.columns)

    if missing:
        raise ValueError(f"Missing required road features: {sorted(missing)}")

    length = pd.to_numeric(roads["length"], errors="coerce")

    if (length.dropna() < 0).any():
        raise ValueError("Road lengths cannot be negative.")

    def optional(name: str) -> pd.Series:
        """Return an OSM column, or aligned missing values if absent.

        Parameters
        ----------
        name : str
            Optional OSM column name.

        Returns
        -------
        pd.Series
            Values aligned with the roads' index.
        """
        return roads[name] if name in roads else pd.Series(np.nan, index=roads.index)

    table = pd.DataFrame({
        "log_length": np.log1p(length),
        "speed_kph": pd.to_numeric(roads["speed_kph"], errors="coerce"),
        "lanes": optional("lanes").map(parse_osm_lanes),
        "width": optional("width").map(parse_osm_width),
        "highway": optional("highway").map(osm_category),
        "surface": optional("surface").map(osm_category),
        "access": optional("access").map(osm_category),
        "junction": optional("junction").map(osm_category),
        "oneway": optional("oneway").map(parse_osm_oneway),
        "bridge": optional("bridge").map(parse_osm_structure),
        "tunnel": optional("tunnel").map(parse_osm_structure),
    }, index=roads.index)

    # Keep all numeric missingness for the imputer; do not fabricate OSM values.
    table = table.replace([np.inf, -np.inf], np.nan)
    return pd.concat([table, extract_geometry_features(roads)], axis=1)


def make_road_preprocessor(
    numeric_columns: Sequence[str] = ("log_length", "speed_kph", "lanes", "width"),
    categorical_columns: Sequence[str] = ("highway", "surface", "access", "junction"),
    boolean_columns: Sequence[str] = ("oneway", "bridge", "tunnel"),
    geometry_columns: Sequence[str] = (
        "mid_lon", "mid_lat", "bearing_sin", "bearing_cos", "straightness",
    ),
) -> Pipeline:
    """Build an unfitted sklearn preprocessor from explicit feature columns.

    Parameters
    ----------
    numeric_columns : Sequence[str]
        Numeric road features (use ``log_length`` for log-transformed length).
    categorical_columns : Sequence[str]
        OSM categorical features to one-hot encode.
    boolean_columns : Sequence[str]
        OSM binary indicators.
    geometry_columns : Sequence[str]
        Projected/geographic features to standardize.

    Returns
    -------
    Pipeline
        Fittable ColumnTransformer with fixed column selection.

    Raises
    ------
    ValueError
        If feature groups contain unsupported or duplicate columns.
    """
    groups = (
        ("numeric", tuple(numeric_columns), NUMERIC_FEATURES),
        ("categorical", tuple(categorical_columns), CATEGORICAL_FEATURES),
        ("boolean", tuple(boolean_columns), BOOLEAN_FEATURES),
        ("geometry", tuple(geometry_columns), GEOMETRY_FEATURES),
    )

    seen: set[str] = set()

    for group, names, allowed in groups:
        if unexpected := set(names) - allowed:
            raise ValueError(f"Unsupported {group} features: {sorted(unexpected)}")
        if len(names) != len(set(names)) or seen.intersection(names):
            raise ValueError("Feature columns must be unique across groups.")
        seen.update(names)

    if not seen:
        raise ValueError("At least one node feature must be selected.")

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            )),
            ("scaler", StandardScaler()),
        ]), list(numeric_columns)))

    if categorical_columns:
        transformers.append(("categorical", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=0.001,
            sparse_output=False,
        ), list(categorical_columns)))

    if boolean_columns:
        transformers.append(("boolean", SimpleImputer(
            strategy="most_frequent", keep_empty_features=True,
        ), list(boolean_columns)))

    if geometry_columns:
        transformers.append(("geometry", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), list(geometry_columns)))

    return Pipeline([("columns", ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0,
    ))])


class SklearnRoadFeatureEncoder:
    """Encode OSM road attributes without coupling them to graph topology."""

    def __init__(self, preprocessor: Pipeline, fit: bool = True) -> None:
        """Configure the sklearn preprocessing pipeline.

        Parameters
        ----------
        preprocessor : Pipeline
            Unfitted training pipeline, or fitted pipeline for another city.
        fit : bool, default=True
            Fit on these roads; False reuses previously fitted statistics.
        """
        self.preprocessor = preprocessor
        self.fit = fit

    @property
    def fitted_preprocessor(self) -> Pipeline | None:
        """Return fitted sklearn preprocessing, when available.

        Returns
        -------
        Pipeline | None
            Fitted pipeline, or None before training fit.
        """
        try:
            check_is_fitted(self.preprocessor)
        except (AttributeError, ValueError):
            return None
        return self.preprocessor

    def encode(self, roads: gpd.GeoDataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode road features and retain unscaled GPS midpoint positions.

        Parameters
        ----------
        roads : gpd.GeoDataFrame
            Ordered road segments with geometry.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Float32 node features [N, F] and float64 GPS positions [N, 2].

        Raises
        ------
        ValueError
            If training numeric road features are entirely unusable.
        """
        table = prepare_feature_table(roads)

        if self.fit:
            columns = self.preprocessor.named_steps["columns"]
            numeric = next((names for kind, _, names in columns.transformers
                            if kind == "numeric"), ())

            for name in ("log_length", "speed_kph"):
                if name in numeric and table[name].isna().all():
                    raise ValueError(f"Required numeric feature entirely missing: {name}")

            encoded = self.preprocessor.fit_transform(table)
        else:
            check_is_fitted(self.preprocessor)
            encoded = self.preprocessor.transform(table)

        array = np.asarray(encoded, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Encoded node features contain nonfinite values.")

        positions = table[["mid_lon", "mid_lat"]].to_numpy(dtype=np.float64).copy()

        return torch.from_numpy(array.copy()), torch.from_numpy(positions)
