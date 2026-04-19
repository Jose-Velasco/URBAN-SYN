from fmm import Network, NetworkGraph, UBODTGenAlgorithm, UBODT, FastMapMatch, FastMapMatchConfig
from typing import Literal


def build_ubodt(
        edges_shp: str,
        ubodt_csv: str,
        delta: float = 0.02,
        mode: Literal["drive", "walk", "bike", "all"] = "drive",
        log_level: int = 2
    ):
    """
    Precompute UBODT in Python
    
    run before map matching. Precompute UBODT.

    refer to official docs form more parameter info:

    https://fmm-wiki.github.io/docs/documentation/configuration/#ubodt_gen
    """
    network = Network(edges_shp, "fid", "u", "v")
    graph = NetworkGraph(network)

    ubodt_gen = UBODTGenAlgorithm(network, graph)
    status = ubodt_gen.generate_ubodt(
        ubodt_csv,
        delta,
        mode=mode,
        binary=False,
        use_omp=True,
        log_level=log_level
    )

    if not status:
        raise RuntimeError("UBODT generation failed")

    ubodt = UBODT.read_ubodt_csv(ubodt_csv)
    return network, graph, ubodt

def build_fmm_model(network, graph, ubodt):
    model = FastMapMatch(network, graph, ubodt)
    return model

def make_fmm_config(
    k: int = 8,
    radius: float = 0.003,
    gps_error: float = 0.0005
):
    """
    f: the number of candidates
    radius: the search radius, in map unit, which is the same as GPS data and network data
    gps_error: the gps error, in map unit
    """
    return FastMapMatchConfig(k, radius, gps_error)