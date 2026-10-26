# NYC MobilityGPT: one-command workflow

Raw NYC → conservative preprocessing → training-only gravity fit → pretraining → DPO → paired generation → notebook-compatible metrics → consolidated report. All NYC code, tests, pins and documentation live here; the original shared model/trainers are reused. See [RESULTS.md](RESULTS.md) for historical outcomes and [HISTORY.md](HISTORY.md) for complete decisions, definitions, accepted defects and core patches.

After installation and input supply, **run the command, then open `report.html`**:

```bash
.venv/bin/python -m nyc_workflow run --input-dir NYC
```

The command prints the absolute page path when finished. Most users need only this one offline page: open it directly in a browser, without a server, internet connection or browser-side installation. It distinguishes training completion from metric availability, compares the fixed one-road/seed-42 pair, explains tradeoffs and provides expandable developer details.

1. Install the environment and this checkout.

   The recorded environment is Python 3.13.5, PyTorch 2.11.0+cu128/CUDA 12.8 and RTX 3080. Pins record that environment; they are not a cross-machine reproducibility guarantee. CUDA and sufficient free memory for the unchanged batch sizes are required. There is no CPU training fallback.

   ```bash
   python3.13 -m venv .venv
   .venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r nyc_workflow/requirements.txt
   .venv/bin/python -m pip install --no-deps -e .
   ```

   The documented editable source installation is the verified execution path. Commands resolve implementation modules through the installed package, not the working directory. Default output is relative to that source package location; any alternative installation must use a writable, unprotected `--output-dir` if its package is read-only or inside an environment. Wheel discovery/build was checked to exclude generated output; a new wheel-installed GPU run was not performed. Do not ship `.venv`, caches, old checkpoints or historical bundles with a source handoff.

2. Supply the original inputs.

   `NYC/` must contain `nyc.geo`, `nyc.rel`, `nyc_mm_train.csv` and `nyc_mm_test.csv`. Geography needs unique nonnegative `geo_id` values and finite lon/lat `coordinates`; relations need known `origin_id`/`destination_id` endpoints and unique `rel_id`. Trip CSVs need unique `traj_id`, comma-separated `rid_list` and aligned nonempty `time_list`. Train/test parent IDs must be disjoint. Conservative repair/splitting is automatic, not a manual input-editing step.

   The original recorded SHA256 identities, useful for checking a supplied copy, are:

   ```text
   7a4e3404844d9e9fb7497c7978300db1f0865ba90f72b374384912a0e75779ce  nyc.geo
   c8ce7c1c4c7d9567f26e96547c737e99dcc81a9be250376547b093863a5e9b4e  nyc.rel
   38497bc3e31736093f2666d2311d5e26d304e15b201846b6a496fd2427be42ff  nyc_mm_train.csv
   e5c3a664a2d31f29171f439ef9e29d27b25bc6aa96fbe41c4a945afddee84e5d  nyc_mm_test.csv
   ```

   ```bash
   sha256sum NYC/nyc.geo NYC/nyc.rel NYC/nyc_mm_train.csv NYC/nyc_mm_test.csv
   ```

   Raw files are read-only workflow inputs, never rewritten or downloaded. Their actual hashes are locked for each new run and checked throughout it.

3. Execute the complete workflow.

   ```bash
   .venv/bin/python -m nyc_workflow run --input-dir NYC
   ```

   The command creates a fresh `nyc_workflow/output/<UTC timestamp>-<nonce>/` directory and prints the absolute `report.html` path at completion. Open that page. After installation and input supply, the workflow needs no historical artifacts, manual handoffs, separate smoke run or confirmation prompts. It automatically runs:

   preflight → prepare → gravity → verify → safety → pretrain → preferences → DPO → select → paired audit → paired generation → canonical metrics → report.

   Preflight checks dependencies, raw structure/geometry/parent separation, configuration, any supplied network and an actual CUDA matrix multiplication before training. The full recipe remains 50,000 candidate roads; 50 safety updates; 3,000 gravity-weighted pretraining updates at batch 64/LR 1e-5; 1,000 original-function preferences; 250 DPO updates at batch 8/LR 1e-5/beta 0.1; selected update 100. Training seed is 160; parent split seed is 42. DPO initializes a distinct frozen reference from this run's fresh pretrained checkpoint. No supervised/PPO stage or old checkpoint substitution occurs.

   Output overrides can be inside or outside the checkout, provided they do not overlap raw inputs, source files, environments or metadata:

   ```bash
   .venv/bin/python -m nyc_workflow run --input-dir NYC --output-dir nyc_workflow/output/my_new_run
   ```

   Existing run directories are never overwritten. The default `output/` tree is Git-ignored; manually chosen output locations elsewhere may need their own ignore rule.

4. Optionally supply the canonical road network.

   ```bash
   .venv/bin/python -m nyc_workflow run --input-dir NYC --canonical-network /absolute/path/nyc.gpkg --network-layer roads
   ```

   Optional `--network-mode-column drive` applies the existing mode filter. The network must exist, declare a CRS and contain usable valid road geometry; invalid supplied networks fail preflight before training. Prepared geography never substitutes for the canonical network.

   When no network is supplied, the pipeline prints a limitation at startup, automatically computes the 11 network-independent metrics and finishes with canonical `status: incomplete`, `complete: false`, and null off-network rates with reasons. The page headline says “Training completed; 11 of 12 metrics available,” explaining the unavailable off-network rate separately. It is not a failure of training or a fabricated score. `--metric-jobs 6` controls bounded CPU metric workers, not scientific budgets/coverage.

5. Optional integration smoke, not a prerequisite for a full run.

   ```bash
   .venv/bin/python -m nyc_workflow run --input-dir NYC --smoke
   ```

   This same command interface executes every stage, including canonical metrics and the final report. Smoke uses two pretraining updates, eight preferences, two DPO updates and selected update 1. It retains the 50-update safety gate, full prepared network, batch sizes, all eligible paired test sections, prompts 1/3 and seeds 42/43/44, and full metric definitions/sampling limits. It can therefore take substantial time. Its scores are integration diagnostics, not full-training evidence or the historical update-100 reporting result.

6. Stop and resume only at a completed stage boundary.

   ```bash
   .venv/bin/python -m nyc_workflow run --input-dir NYC --output-dir nyc_workflow/output/boundary_run --stop-after pretrain
   .venv/bin/python -m nyc_workflow run --input-dir NYC --output-dir nyc_workflow/output/boundary_run --resume
   ```

   Repeat the same `--smoke`, network/layer/filter and metric-worker arguments when applicable. Resume requires the original explicit output path and matching input, source, configuration, interpreter and dependency identities. Each completed artifact is hash-verified before a stage is skipped. Partial/failed stages require a fresh run; this is not optimizer-state continuation. No automatic smaller batch/network, reduced coverage, uniform-gravity fallback or checkpoint substitution is allowed.

   Generated `output/`, environments, caches, build artifacts and run snapshots are excluded from source enumeration. Snapshots cannot recursively ingest themselves or change source identities merely because another run produced files. Code changes, module reorganization and historical path relocation still invalidate strict resumes; never rewrite old manifests to bypass this.

7. Inspect the completed run.

   Open **`report.html`**. Its four sections cover run status, the main comparison, plain-language interpretation, and expandable developer details. Values are rounded for reading; numeric tooltips retain exact saved values. The main pair is always one-road prompt/seed 42, not a best-seed search; all prompt/seed groups are expandable. Existing JS distance and notebook-compatible count-mixing JSD are explained separately.

   The small “What are these files?” guide distinguishes reports/CSVs, model checkpoints, intermediate data/preferences and debugging/reproducibility material. All internal artifacts stay intact; the page links only existing local files and directories. Its styling is embedded and it has no scripts, external assets or automatic network requests.

   `report.md`, `report.json`, the paired report and canonical JSON/CSV remain available for deeper inspection. Keep `pipeline_config.json`, `identity.json`, `preflight.json`, `stage_status.json` and `artifact_manifest.json`. The embedded stage object in `report.json` is a construction-time snapshot; the linked `stage_status.json` is authoritative for final completion. A page written during the final report stage identifies that final sealing follows, without rewriting the journal.

   ```text
   output/<run-id>/
     report.html        # Main offline result; usually all a reader needs
     report.md, report.json, metrics.json, artifact_manifest.json
     prepared/          # Data, mapping, gravity statistics, raw provenance
     safety/, pretrain/ # Fresh training, finite-update checks, strict reloads
     preferences/       # Original-generator dataset and audit
     training/          # DPO milestones/reference-invariance evidence
     model_selected.pt  # -> training/model_step_000100.pt (full recipe)
     evaluation/        # Paired audit, saved routes, all prompt/seed groups
     canonical_metrics/ # Notebook-compatible reports/CSV/JSON, completeness
     source/, logs/     # Supported source snapshot and per-stage diagnostics
   ```

   New checkpoints and saved generations stay in the ignored run because downstream stages and later audits need them. Primary full-training reporting is DPO update 100, one-road prompt, seed 42 with its matching pretrained row; other seeds are supporting evidence, not a best-seed search. Distribution gains can coexist with more revisiting. Coverage exclusions, route overlap, raw repeated observations, supplied prompts and prior test inspection prevent high-fidelity-trip or privacy guarantees.

8. Create an offline page for an existing completed run; recalculate metrics only when intended.

   ```bash
   .venv/bin/python -m nyc_workflow summarize --run-dir /absolute/path/to/completed-run
   ```

   This command only reads saved configuration, selection, stage status and results, then creates `report.html` in that same run. It imports no scientific dependencies, loads no models, and never retrains, generates routes or calculates metrics. It prints the absolute page path and refuses any existing `report.html` (including symlinks); open an existing page instead. Markdown/JSON/CSV reports, checkpoints, snapshots and manifests are not overwritten. Missing optional files or values are labeled unavailable or attributed to their saved consolidated-report fallback, never invented. A failed/partial journal is refused.

   Adding this renderer changes current source identities. Summarizing an old completed run does not restore its strict-resume validity. Preserve original identity/manifests and source snapshots; use matching historical source/path/environment for exact replay, not rewritten records. HTML added afterward is an extra presentation artifact, not retroactively inserted into a sealed old manifest.

   The separate `metrics` command below does calculate metrics from saved routes; it is not needed for HTML generation, and `run` already performs it. Standalone metrics require either an explicit canonical network or explicit permission for incomplete evaluation, and a fresh output:

   ```bash
   .venv/bin/python -m nyc_workflow metrics --run-dir nyc_workflow/output/<run-id> --raw-data-dir NYC --allow-missing-network --jobs 6 --output-dir nyc_workflow/output/metrics_recheck
   ```

   Replace the literal `<run-id>` with the printed run directory. Alternatively pass `--canonical-network /absolute/path/nyc.gpkg --network-layer roads` instead of `--allow-missing-network`. `--validate-only` hashes/checks inputs without calculating metrics; `--benchmark-only` times three complete raw-parent pairs. The notebook is not imported or required; its small provenance record and pure reference tests are packaged here. FastDTW radius 1, raw-parent references and notebook count mixing are preserved, not standard-JSD substitutions. All 12 checkpoint/prompt/seed groups remain separate.

9. Tests and troubleshooting.

   ```bash
   .venv/bin/python -m nyc_workflow check
   .venv/bin/python -m nyc_workflow run --help
   .venv/bin/python -m nyc_workflow metrics --help
   .venv/bin/python -m nyc_workflow summarize --help
   ```

   Invalid inputs/dependencies, nonfinite updates, memory-budget exhaustion, failed audits and missing artifacts stop progression with an error. Once a run has started, inspect `logs/<stage>.log` and its failed stage record; missing paths/configuration may be refused before a run is created. Restore GPU access/free memory and use a fresh run rather than changing the recipe. CUDA-only tests skip without device access; CPU tests do not establish GPU execution.

   Earlier orchestration verification: all 122 regressions passed with GPU access (the CPU-only run passed 121 and skipped one CUDA check). A fresh `run --input-dir NYC --smoke` completed all 13 stages, 36,366 paired records and 12 metric groups, including finite updates, unchanged reference, strict reloads, source/input/artifact hashes, collision refusal and an all-stages-skipped completed resume. Canonical assessment remained incomplete without the network. That GPU execution predates the HTML-only source changes; see [HISTORY.md](HISTORY.md#unattended-one-folder-revision).

   The HTML revision is checked with saved-result fixtures and retained regressions, including full/smoke labeling, exact value agreement, unavailable values, safe escaping, local links, asset isolation and overwrite refusal. A page for the latest existing full run is produced without rerunning scientific work. The latest HTML verification and artifact-preservation findings are recorded in [HISTORY.md](HISTORY.md#offline-results-page-revision).

   Existing external recovery bundles are evidence only, never a runtime prerequisite. They were left untouched by this revision. Do not copy them, environments or generated output into a recommended source handoff; original raw NYC/SF inputs remain supplied separately.
