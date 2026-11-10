"""Diagnostic sparse pilot for weather, mode, temperature, and optional POI metadata."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .conditioning import encode_requests, slice_metadata
from ..models.diff_util import cal_x0_from_noise_ddpm
from .metadata import sha256
from .road_inputs import load_road_inputs
from ..models.factory import make_model
from .sparse_metadata import load_sparse_split


def run_sparse_pilot(dataset, metadata_dir, output):
    """Run one update over known, partial, and all-missing metadata paths."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new smoke-test directory: {output}")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    eids, rates, metadata, rows, manifest = load_sparse_split(dataset, metadata_dir)
    embeddings, graph = load_road_inputs(dataset)
    for length in sorted(eids):
        available = metadata[length]["masks"]
        complete = torch.stack(list(available.values())).all(0)
        candidates = torch.where(complete)[0].tolist()
        weather = metadata[length]["values"]["weather_category"]
        mode = metadata[length]["values"]["transportation_mode"]
        if not candidates:
            continue
        first = candidates[0]
        second = next((i for i in candidates if weather[i] != weather[first]), None)
        third = next((i for i in candidates if mode[i] != mode[first]), None)
        if second is not None and third is not None:
            indices = [first, second, third, first, first]
            break
    else:
        raise ValueError("Need one length bucket with known weather, mode, temperature variation")
    batch = slice_metadata(metadata[length], indices)
    for field in manifest["fields"]:
        name = field["name"]
        batch["masks"][name][-1] = False  # explicit unconditional example
        if name != "transportation_mode":
            batch["masks"][name][-2] = False  # mode-only example
    ids = torch.tensor([eids[length][i] for i in indices], dtype=torch.long)
    values = 2 * torch.tensor([rates[length][i] for i in indices], dtype=torch.float32) - 1
    config = {"channels": 16, "steps": 2, "pre_dim": embeddings.shape[1],
              "residual_layers": 2, "condition_dim": 8, "metadata_fields": manifest["fields"],
              "all_dropout": manifest["training_dropout"]["all_fields_probability"],
              "field_dropout": manifest["training_dropout"]["per_field_probability"],
              "beta_start": 1e-4, "beta_end": .02}
    model = make_model(config)
    encoder = model.diff_model.metadata_encoder
    before = {name: parameter.detach().clone() for name, parameter in encoder.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    losses = model(graph, embeddings, ids, values, batch)
    if not all(torch.isfinite(loss) for loss in losses):
        raise AssertionError("Nonfinite sparse-conditioned loss")
    sum(losses).backward()
    gradients = {}
    for name, module in encoder.encoders.items():
        gradients[name] = sum(parameter.grad.norm().item() for parameter in module.parameters()
                              if parameter.grad is not None)
        if not np.isfinite(gradients[name]) or gradients[name] <= 0:
            raise AssertionError(f"No finite nonzero learning signal for {name}")
    optimizer.step()
    updated = any(not torch.equal(before[name], parameter) for name, parameter in encoder.named_parameters())
    if not updated:
        raise AssertionError("Sparse encoder did not update")
    model.eval()
    null = encode_requests([{}] * len(indices), manifest["fields"])
    with torch.no_grad():
        if torch.count_nonzero(encoder(null, len(indices), torch.device("cpu"), torch.float32)):
            raise AssertionError("All-unknown input did not produce exact-zero conditioning")
        noisy = torch.randn(len(indices), length, embeddings.shape[1] + 1)
        steps = torch.ones(len(indices), 1, dtype=torch.long)
        baseline = model.diff_model(noisy, steps, None)
        if not torch.equal(baseline, model.diff_model(noisy, steps, null)):
            raise AssertionError("Explicit unknown differs from omitted metadata")
        # Hidden placeholder values must never influence predictions.
        masked = copy.deepcopy(batch)
        for name in masked["values"]:
            masked["masks"][name].fill_(False)
            masked["values"][name].fill_(999 if masked["values"][name].dtype == torch.long else float("nan"))
        if not torch.equal(baseline, model.diff_model(noisy, steps, masked)):
            raise AssertionError("Missing values leaked into denoising")
        field_deltas = {}
        for name in batch["values"]:
            isolated = copy.deepcopy(null)
            isolated["values"][name] = batch["values"][name]
            isolated["masks"][name] = batch["masks"][name]
            delta = (model.diff_model(noisy, steps, isolated) - baseline).abs().max().item()
            if delta <= 0:
                raise AssertionError(f"Known {name} did not influence predicted noise")
            field_deltas[name] = delta
        prediction = model.diff_model(noisy, steps, batch)
        shuffled = slice_metadata(batch, list(reversed(range(len(indices)))))
        shuffled_delta = (prediction - model.diff_model(noisy, steps, shuffled)).abs().max().item()
        if shuffled_delta <= 0:
            raise AssertionError("Condition shuffling did not influence predicted noise")

    def continuous_sample(condition):
        return cal_x0_from_noise_ddpm(model.diff_model, model.diffusion_param,
                                      len(indices), length, embeddings.shape[1] + 1, condition,
                                      generator=torch.Generator().manual_seed(123))

    null_sample = continuous_sample(None)
    if not torch.equal(null_sample, continuous_sample(masked)):
        raise AssertionError("Unknown placeholders changed controlled sampling")
    conditioned_sample = continuous_sample(batch)
    if not torch.isfinite(conditioned_sample).all() or torch.equal(conditioned_sample, null_sample):
        raise AssertionError("Conditional samples must be finite and respond to known metadata")
    generated_ids, generated_rates = model.generate_data(
        graph, embeddings, len(indices), length, embeddings.shape[1], batch,
        generator=torch.Generator().manual_seed(123))
    if generated_ids.shape != (len(indices), length) or generated_rates.shape != (len(indices), length):
        raise AssertionError("Incorrect sparse generation output shape")
    if not torch.isfinite(generated_rates).all():
        raise AssertionError("Nonfinite generated rates")
    output.mkdir(parents=True)
    checkpoint = {"architecture_version": 2, "config": config, "metadata_schema_version": 2,
                  "state_dict": model.state_dict(), "metadata_fields": manifest["fields"],
                  "metadata_manifest_sha256": sha256(Path(metadata_dir) / "manifest.json"),
                  "road_mapping_sha256": sha256(Path(dataset) / "extra_file/new2raw_rid.json"),
                  "road_embedding_sha256": sha256(Path(dataset) / "graph/road_embed.txt"),
                  "purpose": "Sparse conditioning smoke test; not trained for conditional fidelity"}
    torch.save(checkpoint, output / "pilot.pt")
    restored = torch.load(output / "pilot.pt", weights_only=True, map_location="cpu")
    reloaded = make_model(restored["config"])
    reloaded.load_state_dict(restored["state_dict"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        if not torch.equal(prediction, reloaded.diff_model(noisy, steps, batch)):
            raise AssertionError("Sparse checkpoint reload changed predictions")
    report = {"status": "passed", "optimizer_steps": 1, "device": "cpu",
              "batch_size": len(indices), "selected_length": length,
              "selected_bucket_indices": indices,
              "training_example_masks": {name: value.tolist() for name, value in batch["masks"].items()},
              "metadata_dropout": {"all_fields": config["all_dropout"], "per_field": config["field_dropout"]},
              "field_gradient_norms": gradients, "encoder_updated": updated,
              "exact_zero_unknown_condition": True, "unknown_placeholder_invariance": True,
              "omitted_equals_all_unknown_predictions_and_samples": True,
              "field_prediction_deltas_from_baseline": field_deltas,
              "condition_shuffling_max_prediction_delta": shuffled_delta,
              "controlled_conditional_sample_delta": (conditioned_sample - null_sample).abs().max().item(),
              "generation_shapes": {"road_ids": list(generated_ids.shape), "rates": list(generated_rates.shape)},
              "checkpoint_reload": "identical predictions",
              "limitation": "Zero missing-field influence is enforced. Learned typical/marginal behavior and conditional fidelity require training and evaluation."}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(output / "generated.npz", road_ids=generated_ids.numpy(),
             rates=((generated_rates.clamp(-1, 1) + 1) / 2).numpy())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "metadata-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_sparse_pilot(args.dataset, args.metadata_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
