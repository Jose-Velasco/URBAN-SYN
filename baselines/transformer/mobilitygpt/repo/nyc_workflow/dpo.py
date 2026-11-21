"""Supported NYC DPO stage adapter, invoked by the workflow pipeline.

Original preference construction, collation, objective, freezing and training
are preserved, with explicit paths, update limits, milestone saves and audits.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from nyc_workflow.paired import fingerprint, write_json
from mobilitygpt.config import get_config_from_args
from mobilitygpt.config_utils import set_seed

from .runtime import require_output, snapshot_sources, source_entries, git_record


def environment():
    free, total = torch.cuda.mem_get_info()
    return {"python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "device": "cuda:0",
            "pre_run_free_bytes": free, "total_gpu_bytes": total,
            "torch_threads": torch.get_num_threads(),
            "numpy": np.__version__, "pid": os.getpid()}


def sources(directory):
    snapshot_sources(directory/'source')
    for name, arguments in [('checkout.diff', ['diff', '--binary']),
                            ('git_status.txt', ['status', '--short'])]:
        with (directory/name).open('x') as stream:
            stream.write(git_record(*arguments))
    write_json(directory/'source_fingerprints.json', {name:fingerprint(p) for name,p in source_entries().items()})


def config_for(directory, samples=100, updates=50, pretrained=None, data_root=None):
    if pretrained is None or data_root is None:
        raise ValueError("Explicit pretrained checkpoint and prepared-data root are required")
    pretrained, data_root = Path(pretrained).resolve(), Path(data_root).resolve()
    config = get_config_from_args(SimpleNamespace(
        dataset="NYCAll", random_trajs=False, lora=False,
        model_path=os.path.relpath(pretrained.with_suffix(""), directory),
        mode="dpo", create_rl_dataset=False, create_dpo_dataset=False,
        dp_training=False))
    config.system.work_dir = str(directory)
    if data_root is not None:
        config.data.root = str(data_root)
    config.training.num_samples = samples
    config.training.learning_rate = 1e-5
    config.training.batch_size = 8
    config.training.num_workers = 0  # DPO loop has no loader; recorded, unused.
    config.training.max_iters = updates  # Also set the existing trainer's hardcoded limit.
    return config


def finite(model):
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def audit(data, base, data_root, pretraining_dir):
    """Describe the original construction without rewriting its tensors."""
    from mobilitygpt.dpo_trainer import (KEY_PCT, KEY_PRT, KEY_CLM, KEY_RLM)
    p = 4
    lookup = {}
    import pandas as pd
    rows = pd.read_csv(data_root / "NYCAll-Taxi/NYCAll_Taxi_trajectory_train.csv",
                       dtype={"traj_id": str, "parent_traj_id": str})
    split = json.loads((pretraining_dir / 'split.json').read_text())
    validation = set(split["validation_indices"])
    for i, sequence in enumerate(base.trajs):
        tokens = tuple(base.stoi[t] for t in sequence)
        lookup.setdefault(tokens, []).append(i)
    eos = base.stoi[base.EOS_TOKEN]
    records = []
    for i, item in enumerate(data):
        chosen, rejected = item[KEY_PCT].tolist(), item[KEY_PRT].tolist()
        cm, rm = item[KEY_CLM].tolist(), item[KEY_RLM].tolist()
        source = chosen[p:]
        matches = lookup[tuple(source)]
        assert cm == [0.] * p + [1.] * len(source)
        assert rm == [0.] * p + [1.] * (len(rejected) - p)
        assert chosen[:p] == source[:p] == rejected[:p]
        assert chosen[:p] == chosen[p:2*p] == rejected[p:2*p]
        records.append({"example_index": i,
                        "possible_source_rows": matches,
                        "possible_source_trajectory_ids": [rows.iloc[j].traj_id for j in matches],
                        "possible_parent_ids": [rows.iloc[j].parent_traj_id for j in matches],
                        "possible_internal_validation": any(j in validation for j in matches),
                        "certain_internal_validation": all(j in validation for j in matches),
                        "outer_prompt_tokens": p, "duplicated_prompt_scored_as_completion": True,
                        "chosen_tokens": len(chosen), "rejected_tokens": len(rejected),
                        "chosen_scored_eos": sum(t == eos and m == 1 for t, m in zip(chosen, cm)),
                        "rejected_scored_eos": sum(t == eos and m == 1 for t, m in zip(rejected, rm)),
                        "chosen_exceeds_context": len(chosen) > 81,
                        "rejected_exceeds_context": len(rejected) > 81,
                        "tensor_devices": {k: str(v.device) for k, v in item.items()}})
    return {"examples": len(records), "duplicated_prompt_examples": len(records),
            "possible_internal_validation": sum(r["possible_internal_validation"] for r in records),
            "certain_internal_validation": sum(r["certain_internal_validation"] for r in records),
            "chosen_truncated_if_sampled": sum(r["chosen_exceeds_context"] for r in records),
            "rejected_truncated_if_sampled": sum(r["rejected_exceeds_context"] for r in records),
            "chosen_scored_eos_counts": dict(Counter(r["chosen_scored_eos"] for r in records)),
            "rejected_scored_eos_counts": dict(Counter(r["rejected_scored_eos"] for r in records)),
            "records": records}


class NonfiniteDiagnostic(RuntimeError):
    pass


def training(policy, reference, config, data, directory, result, updates=50, checkpoint_updates=()):
    from mobilitygpt import dpo_trainer as module
    from finetuners.dpo_finetuning import DPOFinetuner
    initial = {n: p.detach().cpu().clone() for n, p in policy.named_parameters()}
    frozen_reference = {n: p.detach().cpu().clone() for n, p in reference.named_parameters()}
    result.update(updates_completed=0, reference_all_frozen=all(not p.requires_grad for p in reference.parameters()),
                  reference_training=reference.training, actual_device_errors=[],
                  policy_trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad),
                  policy_total_parameters=sum(p.numel() for p in policy.parameters()))
    assert result["reference_all_frozen"] and not reference.training
    assert updates > 0 and all(0 < step <= updates for step in checkpoint_updates)
    assert len(set(checkpoint_updates)) == len(checkpoint_updates)
    assert all(torch.equal(p.detach().cpu(), frozen_reference[n]) for n, p in policy.named_parameters())
    original_init, original_compute = module.DPOTrainer.__init__, module.compute_loss
    original_backward = torch.Tensor.backward
    original_sample = module.random.sample
    original_padded = module.get_padded_batch
    box = []
    sampled = []
    example_indices = {id(item): i for i, item in enumerate(data)}
    log = open(directory / "steps.jsonl", "x", buffering=1)

    def sample(population, count):
        values = original_sample(population, count)
        if population is data:
            sampled[:] = [example_indices[id(item)] for item in values]
        return values

    def padded(examples, max_len, device='cpu'):
        batch = original_padded(examples, max_len, device)
        if "first_batch" not in result:
            result["first_batch"] = {"shapes": [list(t.shape) for t in batch],
                                     "devices_before_compute": [str(t.device) for t in batch],
                                     "dtypes": [str(t.dtype) for t in batch],
                                     "pad_token_id": 1, "padding_loss_mask": 0,
                                     "max_len": max_len}
        return batch

    def compute(*args, **kwargs):
        output = original_compute(*args, **kwargs)
        result["last_loss"] = output[0].item() if torch.isfinite(output[0]).item() else str(output[0].item())
        if not torch.isfinite(output[0]).all():
            result["detection_phase"] = "before backward and optimizer update"
            raise NonfiniteDiagnostic("Nonfinite original DPO loss")
        return output

    def backward(tensor, *args, **kwargs):
        if box and tensor is getattr(box[0], "loss", None) and not torch.isfinite(tensor).all():
            raise NonfiniteDiagnostic("Nonfinite DPO loss before backward")
        return original_backward(tensor, *args, **kwargs)

    def before_step(optimizer, args, kwargs):
        gradients = [p.grad for p in policy.parameters() if p.grad is not None]
        result["last_gradients_finite"] = bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients)
        if not result["last_gradients_finite"]:
            result["detection_phase"] = "after backward, before optimizer update"
            raise NonfiniteDiagnostic("Missing or nonfinite policy gradients")
        assert all(p.grad is None for p in reference.parameters())

    def after_step(optimizer, args, kwargs):
        result["updates_completed"] += 1
        if not finite(policy):
            result["detection_phase"] = "after optimizer update; discard weights"
            raise NonfiniteDiagnostic("Nonfinite policy parameters")
        log.write(json.dumps({"update": result["updates_completed"],
                              "sampled_example_indices": sampled, "loss": result["last_loss"],
                              "chosen_reward": box[0].chosen_reward.item(),
                              "rejected_reward": box[0].rejected_reward.item(),
                              "gradients_finite": True, "parameters_finite": True}) + "\n")
        if result["updates_completed"] in checkpoint_updates:
            step = result["updates_completed"]
            assert all(torch.equal(p.detach().cpu(), frozen_reference[n]) for n, p in reference.named_parameters())
            path = directory / f"model_step_{step:06d}.pt"
            assert not path.exists(), f"Refusing checkpoint overwrite: {path}"
            torch.save(policy.state_dict(), path)
            result.setdefault("milestone_checkpoints", {})[str(step)] = {
                **fingerprint(path), "update": step, "reference_unchanged_at_save": True}

    def init(trainer, *args, **kwargs):
        original_init(trainer, *args, **kwargs)
        trainer.max_iters = updates
        box.append(trainer)
        # Optimizer is created by run(), not __init__; hooks installed below.

    original_optimizer = policy.configure_optimizers

    def optimizer(config):
        value = original_optimizer(config)
        value.register_step_pre_hook(before_step)
        value.register_step_post_hook(after_step)
        return value

    handles = []
    for name, model in (("policy", policy), ("reference", reference)):
        def pre(module, arguments, name=name):
            result.setdefault("forward_devices", {})[name] = {
                "input": str(arguments[0].device), "model": str(next(module.parameters()).device)}
        handles.append(model.register_forward_pre_hook(pre))
    try:
        with patch.object(module.DPOTrainer, "__init__", init), \
             patch.object(module, "compute_loss", compute), \
             patch.object(module, "get_padded_batch", padded), \
             patch.object(module.random, "sample", sample), \
             patch.object(policy, "configure_optimizers", optimizer), \
             patch.object(torch.Tensor, "backward", backward):
            DPOFinetuner(config, policy, reference, data).train()
        result["trainer_iterations"] = box[0].iter_num
    finally:
        log.close()
        for handle in handles:
            handle.remove()
        result["reference_unchanged"] = all(torch.equal(p.detach().cpu(), frozen_reference[n]) for n, p in reference.named_parameters())
        result["reference_gradients_absent"] = all(p.grad is None for p in reference.parameters())
        result["policy_parameters_finite"] = finite(policy)
        result["changed_parameter_names"] = [n for n, p in policy.named_parameters() if not torch.equal(p.detach().cpu(), initial[n])]
        result["frozen_parameter_changes"] = [n for n, p in policy.named_parameters() if not p.requires_grad and n in result["changed_parameter_names"]]
    assert result["reference_unchanged"] and result["reference_gradients_absent"]
    assert result["changed_parameter_names"] and not result["frozen_parameter_changes"]
    assert result["updates_completed"] == updates
    assert set(result.get("milestone_checkpoints", {})) == {str(s) for s in checkpoint_updates}
    path = directory / "model_dpo.pt"
    torch.save(policy.state_dict(), path)
    from mobilitygpt.model import GPT
    reload_model = GPT(config.model)
    loading = reload_model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    result["checkpoint"] = {**fingerprint(path), "strict_cpu_loading": str(loading)}
    for step, metadata in result.get("milestone_checkpoints", {}).items():
        state = torch.load(metadata["path"], map_location="cpu", weights_only=True)
        status = reload_model.load_state_dict(state, strict=True)
        assert all(torch.isfinite(t).all().item() for t in state.values())
        metadata["strict_cpu_loading"] = str(status)
        if int(step) == updates:
            assert all(torch.equal(t, policy.state_dict()[name].cpu()) for name, t in state.items())
            metadata["matches_final_model"] = True


def child(action, directory, artifact, samples=100, updates=50, checkpoint_updates=(),
          pretrained=None, data_root=None):
    if pretrained is None or data_root is None:
        raise ValueError("Explicit pretrained checkpoint and prepared-data root are required")
    pretrained, data_root = Path(pretrained).resolve(), Path(data_root).resolve()
    directory = require_output(directory, [data_root, pretrained])
    directory.mkdir()
    sources(directory)
    result = {"action": action, "status": "failed", "seed": 160,
              "initial_checkpoint": fingerprint(pretrained), "supervised_fallback_used": False}
    started = time.monotonic()
    try:
        assert torch.cuda.is_available(), "CUDA required"
        result["environment"] = environment()
        torch.cuda.reset_peak_memory_stats()
        set_seed(160)
        from datasets.char_dataset import CharDataset
        from datasets.data_utils import load_geo_data, load_trajectory_data, create_dpo_dataset
        from mobilitygpt.trainer_utils import initialize_model
        from mobilitygpt.model import GPT
        config = config_for(directory, samples, updates, pretrained, data_root)
        text = load_trajectory_data(config)
        _, vocabulary, _, relations = load_geo_data("NYCAll", data_root)
        base = CharDataset(config.data, text, vocabulary)
        policy, adjacency = initialize_model(config, base, relations)
        result["policy_initial_strict_load"] = True
        write_json(directory / "config.json", config.to_dict())
        if action == "generate":
            result["generator_training_mode"] = policy.training
            create_dpo_dataset(policy, base, config)
            path = directory / "preference_dataset_dpo"
        else:
            path = artifact.resolve()
        data = torch.load(path, map_location="cpu", weights_only=True)
        assert len(data) == samples
        result["preference_fingerprint"] = fingerprint(path)
        audit_result = audit(data, base, data_root, pretrained.parent)
        write_json(directory / "dataset_audit.json", audit_result)
        if action == "train":
            reference = GPT(config.model, adj_matrix=adjacency)
            reference.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True), strict=True)
            reference.to(config.system.device).requires_grad_(False).eval()
            result["same_shared_adjacency"] = policy.adj_matrix is reference.adj_matrix
            result["reference_initial_strict_load"] = True
            training(policy, reference, config, data, directory, result, updates, checkpoint_updates)
        result["status"] = "passed"
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if isinstance(error, RuntimeError) and "device" in str(error):
            result.setdefault("actual_device_errors", []).append(str(error))
        traceback.print_exc()
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        if torch.cuda.is_initialized():
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json(directory / "result.json", result)
    return 0 if result["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["child"], help="Internal pipeline stage invocation")
    parser.add_argument("--action", choices=["generate", "train"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--checkpoint-updates", type=int, nargs="*", default=[])
    args = parser.parse_args(argv)
    if args.samples < 8 or args.updates < 1:
        parser.error("--samples must be at least 8 and --updates must be positive")
    if len(set(args.checkpoint_updates)) != len(args.checkpoint_updates) or any(
            step < 1 or step > args.updates for step in args.checkpoint_updates):
        parser.error("checkpoint updates must be unique positive values within the update limit")
    if args.action == "train" and args.artifact is None:
        parser.error("--artifact is required for train")
    if Path(args.name).name != args.name:
        parser.error("--name must be a single directory name")
    return child(args.action, args.experiment_dir.resolve() / args.name, args.artifact,
                 args.samples, args.updates, args.checkpoint_updates,
                 args.pretrained.resolve(), args.data_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
