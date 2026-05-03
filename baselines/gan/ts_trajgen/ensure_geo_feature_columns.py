from __future__ import annotations

from pathlib import Path

import pandas as pd
import argparse



DEFAULT_GEO_FEATURES: dict[str, object] = {
    "highway": "unknown",
    "oneway": "unknown",
    "length": 0.0,
    "lanes": "unknown",
    "bridge": "no",
    "access": "unknown",
    "maxspeed": 120.0,
    "tunnel": "no",
    "junction": "no",
    "width": 100.0,
}


def ensure_geo_feature_columns(
    geo_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Ensure a TS-TrajGen .geo file has the road feature columns expected
    by the original Xian preprocessing code (pretrain_gat_fc.py).

    Missing columns are added with default values, while existing columns are
    preserved. This keeps downstream baseline code close to the original.
    """
    geo_path = Path(geo_path)
    output_path = Path(output_path)

    geo_df = pd.read_csv(geo_path)

    for column, default_value in DEFAULT_GEO_FEATURES.items():
        if column not in geo_df.columns:
            geo_df[column] = default_value
        else:
            geo_df[column] = geo_df[column].fillna(default_value)

    geo_df.to_csv(output_path, index=False)

    return output_path

def main() -> None:
    parser = argparse.ArgumentParser(description="TS-TrajGen ensure_geo_feature_columns for pretrain_gat_fc.py")
    parser.add_argument(
        "--geo_path",
        type=Path,
        # default=Path("./data/nyc/nyc.geo"),
        required=True,
        help="Path to *.geo file to ensure a TS-TrajGen .geo file has the road feature columns expected.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        # default=Path("./data/nyc/nyc_features_processed.geo"),
        required=True,
        help="Save path to precessed *.geo file to ensure a TS-TrajGen .geo file has the road feature columns expected",
    )
    args = parser.parse_args()
    geo_path = args.geo_path
    output_path = args.output_path

    ensure_geo_feature_columns(
        geo_path=geo_path,
        output_path=output_path
    )

if __name__ == "__main__":
    main()
