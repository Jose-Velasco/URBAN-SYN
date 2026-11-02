"""Pure conversion helpers copied from the Diff-RNTraj FMM converter.

Source: fmm_scripts/diffrntraj_version/convert_fmm_to_diffrntraj.py.
Kept local so sidecar replay has no runtime dependency on the baseline tree.
"""
import math
import re
from typing import List
import numpy as np

FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
INT_RE = re.compile(r"-?\d+")

def parse_int_sequence(value: object) -> List[int]:
    """Convert FMM's serialized road-ID sequence into Python integers."""
    if value is None:
        return []

    if isinstance(value, float) and math.isnan(value):
        return []

    # Example: "53,53,72,91" -> [53, 53, 72, 91]
    # Repeated roads are intentionally preserved.
    return [int(x) for x in INT_RE.findall(str(value))]

def parse_float_sequence(value: object) -> List[float]:
    """Convert FMM's serialized offset/length sequence into floats."""
    if value is None:
        return []

    if isinstance(value, float) and math.isnan(value):
        return []

    return [float(x) for x in FLOAT_RE.findall(str(value))]

def encode_rates(
    raw_eids: List[int],
    offsets: List[float],
    lengths: List[float],
) -> List[float]:
    """
    Diff-RNTraj moving-rate encoding (inverse of toseq() in model_utils.py):
      first point on a road:    rate = p
      next point on same road:  rate = (p - p_prev) / (1 - p_prev)
    where p = offset / length is the position on the road. The running
    position is tracked with the DECODED value so decoding reproduces the
    clamped positions exactly.
    """
    rates: List[float] = []
    last_eid = None
    last_p = 0.0

    for eid, offset, road_length in zip(raw_eids, offsets, lengths):
        if (
            not np.isfinite(offset)
            or not np.isfinite(road_length)
            or road_length <= 0
        ):
            p = 0.0
        else:
            p = min(1.0, max(0.0, offset / road_length))

        if eid == last_eid:
            if last_p >= 1.0:
                rate = 0.0
            else:
                rate = (p - last_p) / (1.0 - last_p)
                # GPS noise can move the match slightly backwards; Diff-RNTraj
                # cannot represent that, so stay at the previous position.
                rate = min(1.0, max(0.0, rate))
            last_p = last_p + (1.0 - last_p) * rate
        else:
            rate = p
            last_p = p

        last_eid = eid
        rates.append(float(rate))

    return rates
