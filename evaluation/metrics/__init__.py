"""The 12 metrics of the team's spec (Synthetic Trajectory Evaluation Metrics, Oct 2026).

Modules, in the spec's order:
    distribution.py  1-4   geographic, OD flow, trip distance and radius distributions (JSD), plus start and end points
    trajectory.py    5-8   Hausdorff, normalized DTW, EDR and grid-cell Jaccard on paired trajectories
    privacy.py       9-10  membership inference (ROC-AUC) and uniqueness
    road.py          11-12 off-network rate and road connectivity (RSC)

Shared pieces:
    trajset.py   TrajSet, the arrays every metric works on
    grid.py      the shared GPS grid
    jsd.py       Jensen-Shannon divergence (base 2) and histograms
    pairing.py   which generated trajectory is compared with which real one

evaluate.py (one folder up) runs them all and writes the results.
"""

# The metric names used in scores.csv and table.md.
METRICS = [
    "geo_jsd", "od_jsd", "start_jsd", "end_jsd", "distance_jsd", "radius_jsd",  # start/end: the two halves of OD
    "hausdorff_km", "ndtw_km", "edr", "jaccard",
    "mia_auc", "uniqueness",
    "off_network_rate", "rsc",
]
