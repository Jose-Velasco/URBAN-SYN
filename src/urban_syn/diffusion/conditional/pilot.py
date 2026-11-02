"""One CPU optimizer step and controlled conditioning checks on aligned NYC data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..models.diff_module import diff_CSDI
from ..models.diff_util import EDGE_KEY_BASE, cal_x0_from_noise_ddpm
from .metadata import iter_batches, load_aligned_split, sha256
from ..models.model import Diff_RNTraj


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


def encode_weather(categories, vocabulary):
    """Explicit generation input; never sample weather implicitly."""
    missing = [category for category in categories if category not in vocabulary]
    if missing:
        raise ValueError(f"Unknown requested weather categories: {missing}")
    return torch.tensor([vocabulary[category] for category in categories], dtype=torch.long)


def load_road_inputs(dataset):
    dataset = Path(dataset)
    with (dataset / "graph/road_embed.txt").open() as stream:
        count, dimension = map(int, stream.readline().split())
        values = np.loadtxt(stream, dtype=np.float32, ndmin=2)
    if values.shape != (count, dimension + 1) or not np.isfinite(values).all():
        raise ValueError("Invalid road embedding shape/values")
    if not np.array_equal(values[:, 0], np.arange(count)):
        raise ValueError("Road embedding IDs must follow the baseline's zero-based order")
    mapping = json.loads((dataset / "extra_file/new2raw_rid.json").read_text())
    if set(map(int, mapping)) != set(range(1, count + 1)):
        raise ValueError("Road embedding and dataset ID mapping differ")
    embeddings = torch.zeros((count + 1, dimension))
    embeddings[1:] = torch.from_numpy(values[:, 1:])
    edges = np.loadtxt(dataset / "graph/graph_edge.edgelist", ndmin=2)
    edges = edges[edges[:, 2] != 0]
    keys = (edges[:, 0].astype(np.int64) + 1) * EDGE_KEY_BASE + edges[:, 1].astype(np.int64) + 1
    return embeddings, torch.from_numpy(np.unique(keys))


def run_pilot(dataset, metadata_dir, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new pilot output directory: {output}")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    eids, rates, codes, rows, manifest = load_aligned_split(dataset, metadata_dir)
    embeddings, graph = load_road_inputs(dataset)
    # Select real conditions of two distinct observed categories at one exact length.
    for length in sorted(eids):
        first = next((i for i, code in enumerate(codes[length]) if code != 0), None)
        second = next((i for i, code in enumerate(codes[length])
                       if first is not None and code not in (0, codes[length][first])), None)
        if second is not None:
            indices = [first, second]
            break
    else:
        raise ValueError("No length bucket has two observed weather categories")
    selected_eids = {length: [eids[length][i] for i in indices]}
    selected_rates = {length: [rates[length][i] for i in indices]}
    selected_codes = {length: [codes[length][i] for i in indices]}
    ids, raw_rates, weather = next(iter_batches(selected_eids, selected_rates, selected_codes, 2))
    ids = torch.tensor(ids, dtype=torch.long)
    normalized_rates = 2 * torch.tensor(raw_rates, dtype=torch.float32) - 1
    weather = torch.tensor(weather, dtype=torch.long)
    config = {"channels": 16, "steps": 2, "pre_dim": embeddings.shape[1],
              "residual_layers": 2, "weather_categories": len(manifest["vocabulary"]),
              "condition_dim": 8, "beta_start": 1e-4, "beta_end": .02}
    model = make_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    before = model.diff_model.weather_embedding.weight.detach().clone()
    losses = model(graph, embeddings, ids, normalized_rates, weather)
    if not all(torch.isfinite(loss).item() for loss in losses):
        raise AssertionError("Nonfinite forward loss")
    sum(losses).backward()
    gradient = model.diff_model.weather_embedding.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
        raise AssertionError("Weather encoder has no finite nonzero gradient")
    gradient_norm = gradient.norm().item()
    optimizer.step()
    if torch.equal(before, model.diff_model.weather_embedding.weight):
        raise AssertionError("Weather embedding did not update")

    model.eval()
    # Paired comparison holds x_t and t fixed and permutes metadata only.
    noisy = torch.randn((2, length, embeddings.shape[1] + 1))
    steps = torch.ones((2, 1), dtype=torch.long)
    with torch.no_grad():
        prediction = model.diff_model(noisy, steps, weather)
        shuffled = model.diff_model(noisy, steps, weather.flip(0))
    prediction_delta = (prediction - shuffled).abs().max().item()
    if torch.allclose(prediction, shuffled):
        raise AssertionError("Condition-shuffling test failed: predicted noise ignores weather")
    for invalid, expected_exception in [(weather[:1], ValueError), (weather.float(), TypeError),
                                       (torch.full_like(weather, 9999), ValueError)]:
        try:
            model.diff_model(noisy, steps, invalid)
        except expected_exception:
            pass
        else:
            raise AssertionError("Invalid weather input was accepted")

    def sample(condition):
        generator = torch.Generator().manual_seed(123)
        return cal_x0_from_noise_ddpm(model.diff_model, model.diffusion_param, 2, length,
                                      embeddings.shape[1] + 1, condition, generator=generator)

    generated = sample(weather)
    if not torch.equal(generated, sample(weather)):
        raise AssertionError("Sampling did not replay every random draw")
    generation_delta = (generated - sample(weather.flip(0))).abs().max().item()
    if generation_delta == 0 or not torch.isfinite(generated).all():
        raise AssertionError("Conditioned sampling sensitivity/finite check failed")
    generated_ids, generated_rates = model.generate_data(
        graph, embeddings, 2, length, embeddings.shape[1], weather,
        generator=torch.Generator().manual_seed(123))
    if generated_ids.shape != (2, length) or generated_rates.shape != (2, length):
        raise AssertionError("Unexpected decoded generation shapes")
    if not torch.isfinite(generated_rates).all():
        raise AssertionError("Nonfinite generated rates")

    # Save all inference-relevant configuration, including vocabulary and schedule.
    output.mkdir(parents=True)
    checkpoint = {"config": config, "state_dict": model.state_dict(),
                  "metadata_schema_version": manifest["schema_version"],
                  "vocabulary": manifest["vocabulary"],
                  "metadata_manifest_sha256": sha256(Path(metadata_dir) / "manifest.json"),
                  "road_mapping_sha256": sha256(Path(dataset) / "extra_file/new2raw_rid.json"),
                  "road_embedding_sha256": sha256(Path(dataset) / "graph/road_embed.txt"),
                  "purpose": "One-step wiring smoke test; not a trained conditional generator"}
    torch.save(checkpoint, output / "pilot.pt")
    restored = torch.load(output / "pilot.pt", map_location="cpu", weights_only=True)
    reloaded = make_model(restored["config"])
    reloaded.load_state_dict(restored["state_dict"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        if not torch.equal(prediction, reloaded.diff_model(noisy, steps, weather)):
            raise AssertionError("Checkpoint reload changed predictions")
    inverse_vocabulary = {value: key for key, value in manifest["vocabulary"].items()}
    conditions = [inverse_vocabulary[code] for code in weather.tolist()]
    if not torch.equal(weather, encode_weather(conditions, restored["vocabulary"])):
        raise AssertionError("Explicit generation condition encoding changed")
    report = {"status": "passed", "optimizer_steps": 1, "device": "cpu", "config": config,
              "selected_length": length, "selected_bucket_indices": indices,
              "conditions": conditions, "losses": {name: loss.item() for name, loss in
                  zip(("noise", "connectivity", "x0"), losses)},
              "weather_gradient_norm": gradient_norm, "embedding_updated": True,
              "condition_shuffling_max_prediction_delta": prediction_delta,
              "controlled_sampling_max_delta": generation_delta,
              "generation_shapes": {"road_ids": list(generated_ids.shape),
                                    "rates": list(generated_rates.shape)},
              "metadata_shape_dtype_range_checks": "passed",
              "checkpoint_reload": "identical predictions",
              "random_draw_replay": "identical samples for identical conditions",
              "limitation": "Demonstrates alignment and trainable conditioning path, not learned weather fidelity."}
    np.savez(output / "generated.npz", road_ids=generated_ids.numpy(),
             rates=((generated_rates.clamp(-1, 1) + 1) / 2).numpy(), weather=weather.numpy())
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_pilot(args.dataset, args.metadata_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
