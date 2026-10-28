# MobilityGPT

The supported NYC workflow is packaged in [nyc_workflow](nyc_workflow/README.md): raw inputs → preprocessing → gravity-weighted pretraining → DPO → paired evaluation → notebook-compatible metrics and reports.

See the [setup and commands](nyc_workflow/README.md), [historical results](nyc_workflow/RESULTS.md) and [history/core-patch inventory](nyc_workflow/HISTORY.md). Fresh runs default to Git-ignored `nyc_workflow/output/<run-id>/`.

```bash
.venv/bin/python -m nyc_workflow run --input-dir NYC
```

Then open the printed `report.html` path. This self-contained offline page is the main results view. For an existing completed run, use `python -m nyc_workflow summarize --run-dir /path/to/run` to create the page without repeating scientific work or overwriting reports.

Original shared code, SF inputs and supervised/PPO functionality remain. NYC results show distribution/revisiting tradeoffs, not a validated high-fidelity trip generator. Historical evidence is stored separately; see the workflow history for recovery and verification limits.

[License](LICENSE)
