"""Parse selected OpenStreetMap tags from canonical road-network tables."""
import json
import re
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from pint import UnitRegistry
from pint.errors import DimensionalityError, UndefinedUnitError

@lru_cache(maxsize=1)
def _unit_registry() -> UnitRegistry:
    """Reuse one Pint registry for OSM unit conversions.

    Returns
    -------
    UnitRegistry
        Registry used for width conversion to meters.
    """
    return UnitRegistry()


def osm_values(value: Any) -> list[str]:
    """Split an OSM scalar, semicolon list, or serialized JSON list.

    Parameters
    ----------
    value : Any
        Raw OSM tag, possibly missing or multivalued.

    Returns
    -------
    list[str]
        Nonempty, stripped tag values.
    """
    if value is None or value is pd.NA:
        return []

    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]

    if not isinstance(value, (list, tuple)) and pd.isna(value):
        return []

    raw = str(value).strip()

    if not raw:
        return []

    if raw.startswith("[") and raw.endswith("]"):
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            decoded = None

        if isinstance(decoded, list):
            return [str(item).strip() for item in decoded if item is not None and str(item).strip()]

    return [item.strip() for item in raw.split(";") if item.strip()]


def osm_category(value: Any) -> str:
    """Choose the first normalized OSM category or ``unknown``.

    Parameters
    ----------
    value : Any
        OSM tag value.

    Returns
    -------
    str
        Normalized category.
    """
    values = osm_values(value)
    return values[0].lower() if values else "unknown"


def parse_osm_lanes(value: Any) -> float:
    """Return the maximum valid lane count or NaN.

    Parameters
    ----------
    value : Any
        Scalar or multivalued OSM lane count.

    Returns
    -------
    float
        Largest parsed nonnegative lane count, or NaN.
    """
    counts = [float(v) for v in osm_values(value) if re.fullmatch(r"\d+(?:\.\d+)?", v)]
    return max(counts) if counts else float("nan")


def parse_osm_oneway(value: Any) -> float:
    """Map canonical one-way flags to 0/1 with NaN for unknowns.

    Parameters
    ----------
    value : Any
        Boolean or OSM one-way tag.

    Returns
    -------
    float
        One for one-way, zero for two-way, or NaN.
    """
    values = {v.lower() for v in osm_values(value)}

    if values & {"true", "yes", "1", "-1"}:
        return 1.0

    if values & {"false", "no", "0"}:
        return 0.0

    return float("nan")


def parse_osm_structure(value: Any) -> float:
    """Convert a bridge/tunnel tag to an indicator.

    Parameters
    ----------
    value : Any
        OSM bridge or tunnel tag; absent means not explicitly tagged.

    Returns
    -------
    float
        One for an affirmative/typed structure, otherwise zero.
    """
    values = {v.lower() for v in osm_values(value)}
    return float(bool(values - {"no", "false", "0"}))


def _parse_single_width(value: str) -> float | None:
    """Interpret an OSM width value as meters, when possible.

    Parameters
    ----------
    value : str
        One width, optionally with units.

    Returns
    -------
    float | None
        Nonnegative width in meters, or None for invalid input.
    """
    text = str(value).strip()
    if not text:
        return None

    try:
        quantity = float(text)  # Untagged OSM widths are in meters.
    except ValueError:
        match = re.fullmatch(r"(\d+(?:\.\d+)?)'\s*(\d+(?:\.\d+)?)\"?", text)

        if match:
            feet, inches = map(float, match.groups())
            quantity = feet * 0.3048 + inches * 0.0254
        else:
            try:
                quantity = float(_unit_registry().Quantity(text).to("meter").magnitude)
            except (ValueError, TypeError, DimensionalityError, UndefinedUnitError):
                return None

    return quantity if np.isfinite(quantity) and quantity >= 0 else None


def parse_osm_width(value: Any) -> float:
    """Convert scalar or multivalued OSM width tags to meters.

    Parameters
    ----------
    value : Any
        OSM width, potentially JSON list or semicolon-separated values.

    Returns
    -------
    float
        Mean of valid parsed widths in meters, or NaN.
    """
    parsed = [_parse_single_width(text) for text in osm_values(value)]
    valid = [number for number in parsed if number is not None]
    return float(np.mean(valid)) if valid else float("nan")
