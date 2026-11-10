import torch

from .diff_module import diff_CSDI
from .model import Diff_RNTraj


def make_model(config):
    conditioning = ({"metadata_fields": config["metadata_fields"],
                     "all_dropout": config["all_dropout"],
                     "field_dropout": config["field_dropout"]}
                    if "metadata_fields" in config else {"weather_categories": config["weather_categories"]})
    net = diff_CSDI(config["channels"], config["channels"], config["steps"],
                    config["channels"], config["pre_dim"], config["residual_layers"],
                    condition_dim=config["condition_dim"], **conditioning)
    beta = torch.linspace(config["beta_start"] ** 0.5, config["beta_end"] ** 0.5,
                          config["steps"]) ** 2
    alpha = 1 - beta
    return Diff_RNTraj(net, {"T": config["steps"], "alpha": alpha,
                            "alpha_bar": alpha.cumprod(0)})
