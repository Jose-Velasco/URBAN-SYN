# NYC MobilityGPT history and rationale

[README](README.md) contains current installation and execution instructions; [RESULTS](RESULTS.md) is the compact historical report. This history preserves the experiment chronology and distinctions between observed results, accepted limitations and verification of the newly packaged source.

The reporting convention is selected DPO update 100, one-road prompt, seed 42, paired with the matching pretrained row. Seed 42 is fixed for reporting, not the best seed. All prompts/seeds remain evidence. Observed distribution improvements coexist with increased revisiting: this is not a validated high-fidelity trip generator.

## Input problems and conservative export

The original NYC files are observation-level map-matched CSVs, not the repository's blank-line-separated road-sequence text. Original road IDs need consistent remapping. Long consecutive runs of the same road, missing directed transitions and ambiguous gaps prevent direct use as connected training routes. The source training/test parent split must survive cleaning. These are format/connectivity issues; a generated section is not necessarily a whole original trip.

[prepare_nyc.py](preprocessing.py) validates headers, unique parent IDs, nonnegative integer roads, known geography, aligned nonempty timestamps and train/test parent separation. It:

1. Collapses consecutive repeated roads only, retaining half-open zero-based observation ranges and the first timestamp.
2. Inserts one connector only when the original collapsed neighbors have exactly one existing directed two-hop connector. It does not recursively repair gaps.
3. Splits unresolved or ambiguous transitions; discards singleton sections.
4. Ranks roads by distinct training parents, counting each parent once per road, then breaks ties by original road ID. Test trips never select the network.
5. Keeps whole surviving sections and removes candidate roads unused by them. It never truncates to model context 81.

Geography attributes and existing directed relations survive export. New IDs are consecutive in ascending retained original-ID order, and mapping/ranking files make the transformation explicit. Retained train/test CSVs, blank-line text, full per-parent JSONL provenance, metadata and summary preserve discarded sections too. Inserted connectors have null observation ranges and empty timestamps; no timestamp is invented. Observation counts include every source observation represented by a collapsed range. Source inputs are unchanged.

The first NYC20K candidate limit yielded 16,261 used roads, 10,048 training sequences from 9,488 parents, and 2,186 test sequences from 2,079 parents. Training/test parent retention was about 66.2%/58.0%. The historical float32 adjacency had dimension 16,262, storage 1,057,810,576 bytes and measured peak host RSS 6,347,255,808 bytes. Those numbers describe the old allocation, not current boolean storage. Construction alone never proved a training batch would fit.

NYCAll raised the candidate limit to 50,000 without changing cleaning policy. All 43,890 training-used roads fit. It retained 15,524 sequences from 14,237/14,336 training parents (99.31%): 13,443 complete cleaned trips plus 2,081 sections from 794 split parents. There were 84,129 distinct directed relations and 419,553 verified exported transitions. No cleaned training section was lost to network selection.

The fixed training network retains 3,123 test sections from 2,903/3,584 test parents (81.00%): 2,731 complete cleaned trips and 392 sections from 172 split parents. Cleaning discarded 117 singleton sections; fixed-network filtering excluded 734 usable sections and 661 additional parents. Evaluation conclusions do not cover excluded parents. Maximum retained reference length is 124 roads.

## Production fixes retained

Gravity weights previously indexed token windows even though CSV rows represented trajectories, and split weights lost original row indices. [GravityTrajectoryDataset](data.py) makes one index mean one trajectory, then draws a local window within it. Long-sequence tails remain reachable. EOS input padding has ignored targets (-1); the real terminal EOS remains a prediction target. CSV/text order, identity, parent and sequence consistency are checked.

The parent split is shared by gravity fitting and samplers: validation fraction 0.2, shuffled seed 42, floor of parent count. NYCAll has 11,390 training parents/12,436 rows and 2,847 validation parents/3,088 rows. Regional endpoint statistics fit training parents only. The original 1,000 m grid and inverse-square gravity formula, including zero same-region gravity, remain. STRtree accelerates the existing intersection lookup with sorted results.

Training weights remain in full row-index space, held-out weights are zero, and scaling by the maximum preserves positive ratios. Negative/nonfinite weights, row/hash mismatches and all-zero training weights fail; there is no uniform fallback. Zero-weight rows are never drawn. Validation is uniform, deterministic and unweighted. Gravity is a pretraining sampling prior, not preference scoring or learned reward; supervised/DPO/PPO sampling remains unweighted.

The old adjacency path multiplied logits by zero, leaving forbidden probability mass; generation also removed legitimate allowed zero logits. The shared membership mask in [model.py](../mobilitygpt/model.py) now fills forbidden logits with the finite dtype minimum before loss/softmax, preserving all allowed values. Forward and policy use the same rule, before reward slicing. Finite sentinels preserve the original DPO zero-mask gather arithmetic. Decoding consumes the masked output without a second zero-value filter. EOS connectivity remains allowed, including dead ends.

The adjacency builder directly allocates its final boolean matrix on the destination device, avoiding intermediate integer/float/concatenation copies. NYCAll's 43,891-square matrix stores 1,926,419,881 bytes (1.794 GiB). Historical construction measured host RSS 936,873,984 bytes and reserved VRAM 1,950,351,360 bytes. Synthetic regressions compare membership, logits, probabilities, losses and gradients against the previous numeric representation.

The trainer adds deterministic token-weighted held-out evaluation, finite loss/gradient checks and callbacks after actual updates. All full-model pretraining parameters are trainable; configured AdamW weight decay is honored. Non-DP execution avoids unrelated Opacus model rewriting. Existing DP paths remain present; NYC experiments establish no differential-privacy guarantee.

Supervised execution fixes are also retained: move batches to model device, cast chosen/rejected vocabulary tensors to float64 before subtraction, then use stable logsigmoid. Unsupported finetuner constructor arguments were removed in the existing trainer utilities. These fixes do not resolve reward semantics. No original supervised/PPO module was removed or reset.

Optional data-root paths support isolated run-owned prepared exports. These are path adaptations, not new scientific definitions. Unrelated pre-existing edits are preserved; the full precleanup inventory/diff records what was present.

## Experiment history and decisions

The initial 3,000-update NYC20K run used the old masking. Held-out loss fell 9.699673→8.693934, but read-only generation probes found mean routes about 2.3 roads, 218/500 immediate stops and about 99.95% median forbidden probability mass. Additional longer-prompt/temperature probes still stopped early. Replayed sampling exposure suggested concentration, but did not prove a single cause. The masking correction changed the objective; old/new losses are not directly comparable.

A fresh masked NYC20K run started at 1.363691, reached best 1.085993 at update 750 and final 1.091167 at 3,000. Paired one-/three-road validation prompts across seeds 42/43/44 remained too short. Best/final means were 6.222/6.594 roads for one-road prompts versus 21.652 reference; lower validation loss did not establish good continuations. NYCAll was chosen to preserve substantially more training coverage, with fresh initialization because vocabulary dimensions differ.

NYCAll profiling used separate processes, 20 temporary in-memory updates and deterministic validation. The budget was min(85% total VRAM, pre-run free minus 512 MiB), about 8.356 GiB. Batches 8/16/32/64 passed with peak reserved 2.463/3.045/4.227/6.543 GiB; batch 128 failed the budget during validation (11.242 GiB reported). No road coverage was reduced. Batch 64 passed a 50-update gate before full training.

The NYCAll full run used seed 160, 3,000 updates, batch 64, LR 1e-5, zero workers and context 81, evaluating every 250 updates. Validation loss fell 1.449017457→1.091931420 across 76,462 targets/3,094 windows. All 95 best/final state tensors were equal. Peak reserved VRAM was 7,044,333,568 bytes (6.56 GiB). Historical timing fields had a reporting-variable collision; corrected timestamp-based duration is about 206.36 seconds. The later canonical reproduction's timer was corrected; it recorded 204.90 seconds inside the runner and 209.00 seconds including setup/checks.

Gravity exposure is narrower than exported coverage. Zero-weight training rows were 9.1267%; the top 1% carried 17.8011% probability, with effective trajectory count 1,517.86. The historical actual log had 191,340 draws, 6,578 sequences, 6,329 parents and 26,950 roads (61.40% of the retained network); empirical top-1% share was 17.9220%. Holding out parents, zero weights, concentration and finite sampling matter. The snapshot preserves the exact exposure logs.

### Why supervised and PPO are excluded

The unchanged supervised trial first failed because CUDA models received CPU indices, then produced an infinite pairwise loss before backward. The retained device and float64/logsigmoid fixes enabled 10 updates on 100 preferences and later 500 updates on 5,000 preferences. The trainer stopped after one loader pass, never ran validation, and retained a vocabulary-tensor objective rather than a scalar trajectory reward. The final finite loss was about 3.41318e37, dominated by adjacency sentinels. The early callback model.pt differed from the final model_supervised.pt.

The first 5,000-preference generation timed out at 600.2 seconds, last logging 3,951 accepted examples, with no complete artifact. An authorized longer retry completed unchanged generation in 714.7 seconds; the first 100 examples matched the earlier prefix. This execution success did not validate reward quality.

A paired pretrained/supervised generation comparison found 100% immediate stopping for supervised outputs: only the supplied one/three roads remained. Saved-validation loss worsened 1.091931→126.120409. Nominal validation/preference-source overlaps prevented independent reward-validation claims.

The actual unchanged PPO reward path omitted adjacency, left the reward model on CPU, selected the final-position road-token-0 logit and applied sigmoid(logit×0.3). CUDA inputs reproduced a device mismatch; the numerical probe used CPU inputs solely as a diagnostic accommodation. Scores were 2.50e-18–7.98e-18; chosen won 72.8% of 1,000 nominal held-out pairs, but padding changes altered every score and win rates. There were substantial source/parent overlaps. No meaningful scalar trajectory reward, padding invariance or working end-to-end PPO was established.

These experiments explain the DPO-only NYC path. Finite gradients, strict loading and graph validity did not establish better trajectories. Original supervised/PPO code and demonstrated execution fixes remain available; historical probes/checkpoints are preserved for review, not promoted.

### Accepted preference limitations and DPO milestones

The two original generators are different. create_rl_dataset generates two candidates and labels the distance-closest one chosen; its geography lookup counts each road once, undercounting revisits. Ties choose candidate 0; duplicates are accepted; padding/truncation follows labeling. create_dpo_dataset instead chooses the recorded trajectory and rejects one generated candidate; it does not rank two generated routes by distance.

Both retain training-mode dropout, uncapped preference generation, prompt_size=4 including leading EOS (normally three roads), and full prepared training text including internal pretraining validation. Very short sources may put terminal EOS in the prompt. Parent identities are absent from original records; audits preserve all possible source matches for duplicate sequences. DPO retains duplicated outer prompts inside scored completions, chosen/rejected EOS asymmetry, context truncation, road-ID-1 padding and zero completion masks. In the initial 100-example audit, every chosen response scored two EOS tokens while 99/100 rejected responses scored one because sampled terminal EOS was omitted. These accepted methodological limitations were not silently corrected.

The first 100-example DPO diagnostic initialized both policy and frozen reference from pretrained, bypassing supervised fallback. Fifty updates at LR 1e-5/batch 8/beta 0.1 were finite; reference weights stayed bitwise unchanged without gradients; no frozen policy tensor changed. It improved count/distance similarity modestly but increased revisiting.

The intermediate experiment used 1,000 examples and 250 real updates with snapshots 50/100/250, preserving the same original objective, masking, LoRA/output-head freezing and independent pretrained reference. Its first 100 preferences matched the diagnostic prefix. It drew 2,000 pairs, visited 877 records, and used 4.422 GiB peak reserved VRAM. Three chosen examples exceeded context 81 and were sampled seven times; truncation was retained. Of source matches, 169 certainly/242 possibly belonged to internal validation.

Historical validation comparisons (one-road / three-road prompts; seeds pooled):

| Checkpoint | Count JS distance | Distance JS distance | Any revisit |
| --- | --- | --- | --- |
| Pretrained | 0.4383 / 0.4537 | 0.4579 / 0.4691 | 31.33% / 33.33% |
| DPO 50 | 0.4177 / 0.4218 | 0.4460 / 0.4479 | 32.80% / 35.68% |
| DPO 100 | 0.3995 / 0.4074 | 0.4320 / 0.4357 | 34.33% / 36.47% |
| DPO 250 | 0.3667 / 0.3805 | 0.4134 / 0.4107 | 36.93% / 38.75% |

Update 100 was a conservative compromise: closer distributions than 50, less revisiting than 250, which also had two capped samples. Update 50 had lowest revisiting; 250 closest distributions; all three were on the six-metric Pareto frontier. No composite score or loss ranking selected a unique winner, and per-seed reversals remain. The later canonical pipeline locked update 100 before training/test evaluation. Test evaluation never reselected the checkpoint.

All historical milestone/prompt/seed groups remain readable without restoring experiments in milestone metrics (external bundle: `handoff/historical_results/dpo_milestone_metrics.json`), with original selection (external bundle: `handoff/historical_results/dpo_selection.json`). The historical versus canonical comparison (external bundle: `handoff/historical_results/reproduction_comparison.json`) records observed reproduction equality.

### Final evaluation and observed reproduction

Final paired evaluation used all 3,123 retained test sections for one-road prompts and 2,938 for three-road prompts; 185 had no recorded continuation beyond three roads. Seeds 42/43/44 and SHA256-derived per-case seeds pair both checkpoints. Sampling uses eval/inference mode, temperature 1, no top-k, and 81 total tokens including leading EOS: at most 80 roads. Sample-before-cap behavior gives EOS priority when both coincide. Prompts/revisits count in lengths and distances; no failed case is silently dropped.

All 14,336 training and 3,584 test parent IDs were disjoint, including 1,220 possible preference-source parents. Exact route duplication remained: 570 test sections from 522 parents match prepared training routes, and 68 match possible preference sources. Parent separation does not mean route novelty. Internal validation was reused by preference generation, and the final test split had already been examined.

The historical final test and fresh canonical full run both completed 36,366 records/18,183 pairs, zero failures, 72,276 loss targets exactly once in 3,126 windows, including first-road and terminal EOS. All 456,619 probability vectors were finite/normalized with zero forbidden probability. Strict loading covered all 95 state tensors. The canonical evaluation took 3,069.93 seconds (3,072.95 including setup/checks), with peak allocated/reserved VRAM 4,921,582,592/5,198,839,808 bytes.

Supporting pooled results, not the fixed seed-42 primary rows:

| Model | Prompt | Count JS | Distance JS | Revisit | Immediate stop |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained | 1 | 0.3860 | 0.4051 | 31.01% | 5.98% |
| DPO 100 | 1 | 0.3370 | 0.3794 | 34.36% | 5.38% |
| Pretrained | 3 | 0.3980 | 0.3889 | 30.96% | 7.43% |
| DPO 100 | 3 | 0.3441 | 0.3625 | 34.31% | 6.61% |

Unweighted test loss was 1.093343127→1.084189513. Mean one-road lengths were 12.084→13.503 roads versus 22.143 references; meters 3,290.2→3,555.5 versus 7,545.0. Real-section revisit rates were only 3.23%/3.34%. Longer outputs can be looping; marginal distribution similarity does not validate continuations. Rare caps remain and cannot cover reference tails beyond 80 roads.

The fresh canonical export and pretrained/selected checkpoint hashes matched historical artifacts; pooled metrics also matched exactly. That was observed after the run, not an enforced equality threshold, a tuning target or a cross-machine guarantee. Historical restart/overwrite checks passed with then-matching sources. The smoke passed the same coverage/audits with reduced training and is integration evidence only.

## Fixed primary reporting rows

The primary report is update 100, one-road prompt, seed 42 with its matching pretrained baseline. Other seeds and prompts remain supporting evidence in the external machine-readable reports; seed 42 was not ranked or chosen for better scores.

| Model, prompt 1, seed 42 | Count JS distance | Meter-distance JS distance | Any revisit | Immediate stop |
| --- | ---: | ---: | ---: | ---: |
| Pretrained | 0.388136 | 0.413802 | 31.03% | 5.92% |
| DPO update 100 | 0.340806 | 0.388197 | 34.23% | 5.35% |

These are 3,123 paired cleaned sections per row. The raw-reference notebook-compatible calculation uses other populations and definitions: trip-distance count-mixing divergence is 0.233230→0.212182 and radius divergence 0.200423→0.187795, while geographic and OD values slightly worsen (0.316749→0.319106 and 0.647332→0.647560). Off-network values remain unavailable without the canonical network. This mixture of outcomes and increased revisiting does not validate high-fidelity trips.

## Raw-reference notebook-compatible metrics

[metrics.py](metrics.py) is independent of historical evaluators and does not execute/import the notebook. It reverses generated IDs once through the explicit mapping, computes [latitude, longitude] representative coordinates as the arithmetic mean of geometry endpoints, and keeps every generated occurrence, prompt, immediate stop and cap.

Distributions and privacy nonmembers use all original raw test trips; privacy members use all original raw training trips. Consecutive raw observations remain, unlike cleaned paired reports. Pairwise references are complete raw parents joined through explicit section provenance; multiple generated sections can share a parent. There is no suffix-ID heuristic, OD matching, truncation or file-order pairing.

| Metric | Exact convention |
| --- | --- |
| Geographic count-mixing JSD | In-bounds visit counts on 87×90 grid, 0.005° cells from lon -74.2963164, lat 40.4983251; index x×90+y |
| OD count-mixing JSD | Joint first/last in-bounds grid-cell counts |
| Trip-distance count-mixing JSD | Consecutive Haversine km sum; 1,000 bins to 40.71 km plus overflow |
| Radius count-mixing JSD | Mean Haversine km to arithmetic coordinate mean, not RMS radius; 100 bins to 10.6 km plus overflow |
| Hausdorff | Symmetric maximum nearest-point Haversine distance, km |
| Normalized DTW | FastDTW radius 1 Haversine cost divided by path length, km |
| EDR | Edit distance normalized by max sequence length; match strictly below 100 m |
| Grid Jaccard | Unique visited in-bounds grid-cell intersection/union; empty union undefined |
| MIA AUC | Average-rank AUC using negative nearest-generated mean matched-point distances |
| Uniqueness | Fraction strictly more than 50 m from nearest sampled training trajectory |
| Off-network | Representative points strictly more than 30 m from projected canonical roads |
| RSC | Original directed .rel transition validity; identical consecutive IDs accepted; singletons contribute zero transitions |

Earth radius is 6,371 km. Floor/index grid bounds extend beyond the notebook's plotting maxima (-73.868091, 40.9410935); excluded out-of-grid points/ODs are counted. Pairwise evaluation takes up to 5,000 sorted pairs with seed 42 and per-metric nanmean. Privacy independently samples up to 2,000 items with seeds 43/44/45, interpolates every route to 32 points, uses blocks of 32 queries and float32 distance storage. Every generated section of a queried test parent is excluded from its nearest-generated pool; identical routes from other parents remain. No remaining candidate makes that score and AUC unavailable rather than dropping the query.

The notebook's function named JSD mixes raw counts first: R=(p+q)/sum(p+q), then 0.5 KL(p/sum(p)||R)+0.5 KL(q/sum(q)||R), using natural logs. Standard equal-weight JSD instead mixes the independently normalized distributions. Unequal count totals therefore change the notebook quantity; standard JSD bounds cannot simply be assumed. This behavior is preserved and labelled, not scientifically redefined. Earlier paired metrics use square-root base-2 JSD distance on occurrence-meter histograms; they are different quantities and populations.

The original notebook hash and extraction cells are packaged in [notebook_provenance.json](notebook_provenance.json), with the former version preserved unchanged at `handoff/notebook_provenance.json` in the external bundle; frozen pure definitions support independent parity tests in `tests/notebook_metrics_reference.py`. The complete notebook remains in the snapshot. No notebook is needed for metric execution or retained tests.

The full saved CPU metric evaluation took about 372 seconds with six workers, covered 12 checkpoint/prompt/seed groups and 36,366 records, and preserved audited hashes. All 11 network-independent metrics were available. The canonical network was not supplied: every off-network score is null with a reason, status incomplete, complete=false. Prepared geometry must never masquerade as the missing canonical network. A supplied canonical network needs declared CRS, valid geometry, projected nearest-road distances and any explicit road-mode filtering.

The geometric nearest-neighbor attack is not a privacy guarantee; representative-point geometry, interpolation, raw/section population differences, shared parents, conditional prompts and previously examined test data limit interpretation. No composite winner is declared. Compact results (external bundle: `handoff/final_results.md`) retain the primary pair and link all supporting groups.

## Mapping findings and examples

The archived offline viewer joined generated remapped IDs directly to prepared geography; reverse mapping was for original-ID labels. It kept occurrence order and separate road geometries without simplification, reorientation, invented connectors or set-based route reconstruction. Repeated-road overlays exposed loops hidden by overlapping lines. Graph direction and stored geometry orientation are separate concepts.

Coordinates were finite NYC lon/lat within [-74.2889775,40.4997025,-73.8895793,40.9226594]. WGS84-compatible degrees were inferred from extent; the CSV did not certify a CRS/datum. Display used a local equirectangular projection with cos(40.75°) longitude scaling; evaluation distances came from saved road lengths, not display coordinates. The viewer verified 18,183 paired keys, 36,366 records and 548,330 directed pairs, with no missing geometry/reference or recorded endpoint mismatches in this build. Browser checks ran offline, with Playwright used for testing only.

Gallery selection was the first matching lexicographic trajectory-ID, numeric prompt-length, numeric seed key per category, not a representative sample or a visual ranking. Examples (section, prompt, seed):

| Category | Illustrative key | Matching paired cases |
| --- | --- | ---: |
| DPO-only revisiting | 10036:0, 1, 43 | 674 |
| Pretrained-only revisiting | 10059:0, 1, 43 | 65 |
| Both revisit | 10032:0, 1, 44 | 5,569 |
| Neither revisits | 10017:0, 1, 42 | 11,875 |
| Pretrained immediate stop | 10017:0, 3, 44 | 1,215 |
| DPO immediate stop | 10017:0, 3, 44 | 1,087 |
| Pretrained cap | 11288:0, 1, 44 | 4 |
| DPO cap | 11288:0, 1, 44 | 11 |

The default explorer key was 10017:0,1,42. The recorded route is one possible continuation, not unique ground truth for an origin. Frequency layers included prompts/revisits and averaged model seeds when normalized per 1,000 eligible sections; raw visits were not an independent demand-accuracy measure. All maps excluded roads/parents outside training-network coverage. Snapshot members visualizations/nyc_pipeline_full_20261020/interpretation.md, gallery_routes.json, example_selection.json, verification.json and browser_checks/ preserve exact examples, figures and checks.

## Shared core patches

The model and trainers remain in the original packages. The NYC package calls those implementations; it does not duplicate the architecture or optimization code.

| Shared file | Necessary retained changes and reason |
| --- | --- |
| `datasets/data_utils.py` | Optional explicit data root; direct boolean adjacency with EOS. Original preference generators and labels remain unchanged. |
| `mobilitygpt/model.py` | Shared finite membership mask in forward/policy, consistent decoding, configured AdamW weight decay; checkpoint schema unchanged. |
| `mobilitygpt/trainer.py` | Deterministic token-mean validation, finite loss/gradient checks, actual-update callbacks and DP-only model rewriting. |
| `mobilitygpt/trainer_utils.py` | Trajectory-indexed gravity sampling, parent-separated holdout, preserved weight ratios, no gravity in later stages, removal of unsupported finetuner arguments. The existing initializer moved here unchanged from `main.py` so installed code can reuse it. |
| `mobilitygpt/reward_trainer.py` | Device transfer and float64 subtraction/stable logsigmoid for demonstrated execution failures; vocabulary reward semantics remain unvalidated. |
| `preprocessing/gravity_based_sampling.py` | Training-parent-only fit, explicit root, metadata hashes and sorted STRtree intersection lookup. The original formula/grid remain. Imports the relocated trajectory split helper. |
| `main.py` | Re-exports the same `initialize_model` helper from shared trainer utilities; original entry point and supervised/PPO paths remain. |
| `pyproject.toml`, `preprocessing/__init__.py` | Discover/install the NYC, dataset and preprocessing packages and ship requirements/provenance/docs; no import-path hacks. |

All existing unrelated edits are preserved. `data.py` is the relocated NYC trajectory adapter formerly in `datasets/gravity_dataset.py`; the upstream shared trainer imports it directly. The pure paired protocol lives in `paired.py`, while preprocessing, pretraining, DPO, evaluation and metrics remain separate modules. Child stages use installed `-m nyc_workflow.<module>` commands with absolute run-local paths. Source snapshots work without Git metadata; when available, Git diff/status are additional records.

## Cleanup and external relocation history

The first archive was `archives/nyc_history_20261020/historical_evidence.tar.gz`: 738 members, SHA256 `2ddffb2af5b199593cf50ef7b51abea10cd0a48198db1bb337bf97df90f0639b`. It predates the newest metric/viewer work and contained an unexecuted 79-file proposal, not a cleanup receipt.

The extension `archives/nyc_handoff_20261021/precleanup_snapshot.tar.gz` preserved 1,222 hash/size/symlink-verified members, including all later evaluator, notebook, viewer, experiment and documentation versions. Its SHA256 is `aa22148f770865ea956b9d7e4ef43dadfed3fd9240d6c11408674afdaf906f9e`. After the user explicitly approved a new exact manifest, 973 entries totaling 1,154,964,488 bytes were removed individually with zero failures. The manifest hash remains `07aa790edf977568c1ededbe846891a71333aaef6a291715c71812ae0d5a19cf`; the per-file receipt/journal remain with the evidence. Original inputs, usable checkpoints, saved canonical generations, upstream code and source snapshots were retained.

For this source-only packaging, the separate external destination is `../MobilityGPT-results/historical_bundle_20261021_source_only/`. It preserves repository-relative paths for `runs/`, `evaluation_outputs/`, `handoff/`, both archive directories under `archives/`, supporting `evidence/`, the full former root documentation and a current source snapshot. The external `_relocation/manifest.json` lists each copy/removal path, size, content hash, symlink and target. The initial external copy preserved 331 files/1,457,549,695 bytes and verified every content hash/size and symlink target before any removal. The user then separately approved this exact manifest (SHA256 `6f75659992251e8cb101c51c7cdb35707a2b5e2df92336aa27c35b37d5a5b1a8`): 300 original files/1,449,858,070 bytes and 189 directories. Every approved file was removed individually, then every listed directory only when empty, with zero failures and no unlisted removals. All external replacements and protected inputs reverified afterward. `_relocation/receipt.json` is the completed receipt; `removal_journal.jsonl` records each outcome. The immutable proposal and earlier copy/verification receipts preserve their original status; the completed removal receipt is authoritative. The previous 973-file approval was not reused for relocation.

Raw NYC inputs, original SF data, `.git`, environments and unrelated files were excluded from relocation. Both full documentation versions remain unchanged as `NYC_REPRODUCTION_GUIDE.md` and `NYC_PREPROCESSING_AND_PIPELINE_NOTES.md` in the external bundle; earlier versions also survive in the verified archive. The then-two package documents preserved their useful findings without requiring local experiment folders; the later one-folder revision adds the compact `RESULTS.md`.

For that relocation, the source handoff was the 61-file export recorded by `_relocation/post_removal/source_handoff_manifest.json`, with per-file sizes and SHA256 values and an explicit inclusion/exclusion policy. It included the then-current NYC package/tests/docs/pins/provenance, original shared source and upstream project files. Raw inputs, `.git`, environments, caches, build metadata, historical results, archives and legacy NYC modules/notebook/viewer assets were excluded from that export, not silently erased locally. That historical export remains preserved unchanged as `_relocation/post_removal/source/`; it is not the subsequent unattended revision. Separate raw inputs and external evidence are supplied only as needed.

To restore evidence, use a fresh directory and copy selected members from the bundle with symlinks preserved, including each `resolved_target` in the external manifest. To inspect the archived viewer, run these commands from the external bundle directory:

```bash
tar -xOzf archives/nyc_handoff_20261021/precleanup_snapshot.tar.gz visualizations/nyc_pipeline_full_20261020/interpretation.md
restore_dir=$(mktemp -d /tmp/mobilitygpt-restore.XXXXXX)
tar -xzf archives/nyc_handoff_20261021/precleanup_snapshot.tar.gz -C "$restore_dir" visualizations/nyc_pipeline_full_20261020
```

Archive members retain repository-relative paths. Never extract over current work. The original raw inputs must be supplied separately. Recorded manifests, checkpoint hashes and absolute paths are immutable. Exact historical replay or strict resume can require the original source tree, interpreter/environment, path identities and archived intermediates; moving files and reorganizing modules invalidates current-source strict resumes. Do not rewrite historical identities to imply portability. New commands instead take explicit saved-run/raw-data paths and verify content identities when auditing relocated artifacts.

## Verification of this organization

Verification uses an isolated source export with no historical results, archives, notebook, old diagnostic modules or `handoff/` directory. It tests wheel installation and imports from an unrelated directory, CLI help, the retained regressions, synthetic module-stage handoffs, safe boundary restart/refusal, explicit external paths, CPU loading/auditing of the externally copied checkpoints and data, and notebook-free validation of all 12 saved metric groups. Raw input and preserved evidence hashes are checked. Detailed results are kept externally under `_relocation/verification/`.

The installed wheel passed 111 regressions: 110 passed and one CUDA-only supervised-device check skipped. Five command-help checks and installed imports passed with Python's isolated mode from an unrelated directory. Both externally stored checkpoints strictly loaded all 95 state tensors; their hashes remained unchanged. The CPU paired audit verified 3,123 sections/2,903 parents and 72,276 targets in 3,126 windows. All 12 saved metric groups validated without the notebook or `handoff/`. Raw NYC/SF identities and all 331 externally preserved file hashes matched. Scientific helper AST comparisons and byte comparisons of the relocated adapter/paired modules passed. A synthetic handoff fixture caught and resolved the initial output-guard rejection of the pipeline's own external-run evaluation subdirectory.

After the approved originals were removed, the actual source-only checkout again passed all 111 tests (110 passed, one CUDA skip). The final post-removal source/export hashes, documentation-link checks, CLI checks and protected-input/evidence checks are recorded separately under `_relocation/post_removal/`; earlier verification records remain unchanged. Only documentation describing completed removal changed after the installed-code verification.

For the preceding source-only relocation, no full NYC training, preference generation or trajectory generation was rerun. Synthetic regression updates exercised tiny fixtures only. Historical RTX 3080 execution was historical evidence, not verification of that new package's full GPU workflow. CUDA skips and the missing canonical network were disclosed; neither was presented as a completed evaluation.

## Unattended one-folder revision

The subsequent requested interface is `python -m nyc_workflow run --input-dir NYC`. The earlier external-output-only restriction is superseded: fresh UTC timestamp/nonce directories default to Git-ignored `nyc_workflow/output/`; explicit internal or external overrides are allowed when they do not overlap raw inputs, source files or environments/metadata. Existing directories still refuse overwrite, and resume still requires an explicit matching original run path. No existing recovery bundle was modified or deleted by this revision, and no new sibling results directory is created.

Configuration schema 3 adds a preflight stage before preparation and notebook-compatible canonical metrics after paired generation, before the consolidated report. Preflight validates dependencies, configuration, raw geometry/relations/parent IDs/timestamps, an actual CUDA operation and any supplied canonical network. A supplied missing/invalid network is an error before training, never an excuse to use the no-network policy. Omitting the network intentionally completes the 11 network-independent metrics, prints the limitation at startup and in final reports, records null off-network values and marks canonical assessment incomplete. Standalone metric commands retain their explicit incomplete-evaluation permission requirement.

The complete chain is preflight → prepare → gravity → verify → safety → pretrain → preferences → DPO → select → paired audit → paired generation → canonical metrics → report. Scientific budgets, seeds, training-only gravity, preference defects, independent frozen reference, milestone selection, full paired populations and metric definitions are unchanged. Smoke remains 50 safety updates, two pretraining updates, eight preferences, two DPO updates and selected update 1, with full network/paired/metric coverage. It is optional for users, not a prerequisite or manually connected workflow.

Source enumeration is restricted to supported package `.py` files and importable subpackages plus pins/provenance/project metadata. It prunes `output/`, environments, caches, build directories, symlink directories and marked run roots. Generated snapshots cannot recursively copy themselves or enter source identities. Run-owned `canonical_metrics/` may live inside its own input run, but no arbitrary input-run subdirectory may be used as metric output; outputs remain fresh and protect raw inputs/network files. Stage journals, exclusive artifact creation, locked identities, strict checkpoint loads, reference invariance, finite/memory checks and completed-boundary restart refusals remain active. Source changes invalidate previous strict-resume identities; recorded source snapshots and all historical identities remain unchanged.

`reporting.py` consolidates both evaluator results and links the selected checkpoint, matched baseline, paired report and canonical report/JSON. `preflight.py` contains validation, keeping the orchestration, stage implementations and reporting separate. Necessary shared upstream patches listed above are unchanged by this revision. [RESULTS.md](RESULTS.md) now keeps compact historical tables in the source package without requiring the evidence bundle.

New regressions cover default output/collision refusal, input/source/symlink overlap, generated-source exclusion, full stage ordering including metrics, no historical handoffs, missing versus invalid network behavior, safe run-owned metric output, immediate failure stopping and completed-stage resume. Before the GPU smoke, 122 CPU regressions ran: 121 passed and one CUDA-only check skipped in the sandbox. The completed fresh one-command smoke verification is recorded below; historical full-budget results are not substituted for it.

The fresh command `.venv/bin/python -m nyc_workflow run --input-dir NYC --smoke` completed without manual stage handoffs in `nyc_workflow/output/20261021T215626.089423Z-3a96ba/`. All 13 stages passed. The 50-update safety gate and two fresh pretraining updates were finite, and strict pretraining reload reproduced exact logits. Eight preferences fed two DPO updates; the separate reference remained bitwise unchanged with no gradients, frozen policy parameters stayed unchanged and both milestone checkpoints strictly reloaded. Selection used smoke update 1, not the historical/full-budget reporting update 100.

The GPU-enabled paired audit ran all 122 regressions with no skips. Both evaluator checkpoints strictly loaded 95 state tensors. New generation produced 36,366 records/18,183 complete pairs, zero failed/dropped cases, 72,276 targets exactly once in 3,126 windows, and 107,774 checked probability vectors with finite normalized probabilities and zero forbidden mass. Paired evaluation took 805.72 seconds; the automatically chained CPU metric stage took 297.03 seconds. All 12 checkpoint/prompt/seed groups have 11 finite network-independent metrics. Off-network values are null with reasons and the canonical assessment is explicitly incomplete; a real canonical-network assessment was not performed. Synthetic fixtures verify its validation/CRS/filter/distance contracts.

Post-run verification rechecked every stage output, both pretrained checkpoints, the selected symlink, raw inputs, all 55 recorded source identities and the sealed artifact manifest. A completed-run `--resume` skipped all 13 stages without scientific execution or journal/artifact changes. A non-resume attempt at the same output refused collision without changing artifacts. All final report links resolve. Derived verification is saved as `smoke_verification.json` inside that ignored run, separate from the original sealed artifact manifest. The report's embedded stage snapshot is taken during its own construction; `stage_status.json` is the authoritative final journal.

Editable installation and isolated-directory CLI help were checked. A 65-file source-only export, without data/results/archives/notebook, built a wheel with only the six intended packages and the three documents/pins/provenance; discovery/build excluded the real generated output tree. That wheel build is packaging evidence, not a separately executed wheel-installed GPU workflow. The new smoke used the documented editable source installation. Executable source was frozen throughout it; only completion documentation changed afterward. No new full-budget run, checkpoint retuning, scientific redefinition, external-bundle mutation or unrequested deletion was performed. Historical full-training quality findings remain exclusively the historical evidence in [RESULTS.md](RESULTS.md).

## Offline results page revision

The user-facing result is now `report.html` in the existing run directory. The normal final-report stage produces it alongside the retained Markdown/JSON reports, includes it in new-run artifact inventories and prints its absolute path at command completion. No new output hierarchy, large-artifact copies or third-party browser assets are introduced. `html_report.py` uses only the Python standard library; it reads saved configuration, selection, final stage status and evaluator JSON, not models, trajectories, raw inputs or the notebook. It does not calculate scientific metrics.

The page has four sections: run status, fixed main comparison, plain-language interpretation and expandable developer details. It distinguishes full training from smoke, identifies the configured selected checkpoint and matching pretrained baseline, and headlines missing-network results as “Training completed; 11 of 12 metrics available.” Null off-network values and their saved reasons remain explicit. Primary reporting is one-road prompt/seed 42, not the best seed; all saved checkpoint/prompt/seed groups remain expandable. Display values are rounded for readability, with exact saved numbers in tooltips and machine-readable cell attributes. Percentage conversion is presentation only.

Readable names, units and interpretation labels distinguish cleaned-section base-2 square-root JS distance from the raw-reference notebook count-mixing divergence. Interpretation compares the saved values without reranking seeds or calculating a composite winner. Distribution changes, revisiting, coverage/overlap, raw repetitions, previous test inspection and geometric-privacy limitations remain explicit; no significance, validated trip fidelity or privacy guarantee is asserted. The file guide separates result reports/CSVs, loadable checkpoints, intermediate data/preferences and debugging/reproducibility files. Most readers need only the HTML page; optional file navigation requires the surrounding run folder.

The page embeds its CSS and all displayed data, with no scripts, external fonts/images/stylesheets or automatic resource requests. Native HTML details/summary controls work offline. All saved text and configuration are HTML-escaped; artifact URLs are encoded and constrained to existing run-local files/directories, including valid symlink targets. Missing optional files are labeled unavailable or attributed to saved consolidated-report fallbacks. During normal final-stage rendering, completion refers to the already completed training/evaluation stages; the page notes that final journal/artifact sealing follows. It does not preemptively mark the journal complete or link a sealing artifact that does not yet exist.

`python -m nyc_workflow summarize --run-dir <existing-run>` creates only a new `report.html`, prints the absolute path and refuses any existing destination, including dangling symlinks. A failed/partial stage journal is refused. Saved Markdown/JSON/CSV, checkpoints, prepared data, preferences, logs, snapshots, manifests and verification records are never overwritten. The original sealed manifests are not extended retroactively to claim that a newly added presentation artifact existed at training time. These source changes invalidate prior current-source strict-resume identities; use the recorded source/path/environment for exact replay, not rewritten identities. Summarization is not stage resume or renewed scientific verification.

An already completed full-budget run was found at `nyc_workflow/output/20261021T222805.360824Z-3d4205/`, with `smoke: false`, all 13 journal stages complete and selected update 100. This revision ran the new summarize command on that run, producing its 95,422-byte page; it did not start or repeat that full training, paired generation or metric evaluation. The page shows all 12 saved canonical metric groups, with 11 available metrics per group and off-network values unavailable. All 266 displayed numeric cells (including the repeated primary comparison) exactly match their saved JSON values, and all 32 page-section/local-artifact links resolve.

Verification ran 140 regressions: 139 passed and one CUDA-only device test skipped in the sandbox. Eighteen HTML-specific tests cover full/smoke labels, selected checkpoint, fixed reporting choices, complete versus missing-network assessment, exact numeric agreement, missing/nonfinite values, absent optional files, HTML escaping, URL restrictions, no external resources, lazy scientific-dependency-free CLI imports, overwrite refusal and final-report-stage integration with saved fixtures. No new NYC smoke/full run was needed for this presentation-only revision. Before/after SHA256/size/symlink inventories of both current runs confirmed all 784 pre-existing files unchanged, including checkpoints, sealed manifests and verification records; the only added run file is the full run's `report.html`. Actual overwrite refusal was also checked. No browser binary/automation was available, so offline checks used HTML parsing, resource/CSP inspection and local-link validation rather than a live-browser or visual test. Existing external recovery bundles were left untouched.
