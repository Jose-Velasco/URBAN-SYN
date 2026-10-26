# Historical NYC results

These tables describe the recorded full-budget NYCAll run, not the new orchestration smoke. [README.md](README.md) explains execution; [HISTORY.md](HISTORY.md) preserves definitions, chronology, mapping findings, failures, fixes and evidence/restoration locations.

## Fixed primary reporting pair

Selected DPO update 100, one-road prompt, seed 42, compared with the matching pretrained baseline. Seed 42 is a fixed reporting convention, not the best seed. Each row covers 3,123 paired cleaned test sections.

| Model, prompt 1, seed 42 | Count JS distance | Meter JS distance | Any revisit | Immediate stop |
| --- | ---: | ---: | ---: | ---: |
| Pretrained | 0.388136 | 0.413802 | 31.03% | 5.92% |
| DPO update 100 | 0.340806 | 0.388197 | 34.23% | 5.35% |

Update 100 was locked before canonical training/test evaluation, based on the earlier validation tradeoff: update 50 revisited less; update 250 matched length/distance distributions more closely but revisited more. No composite score established a unique best model.

## Supporting pooled paired results

These aggregate seeds 42/43/44; they are not the primary seed-42 rows.

| Model | Prompt roads | Count JS distance | Meter JS distance | Any revisit |
| --- | ---: | ---: | ---: | ---: |
| Pretrained | 1 | 0.3860 | 0.4051 | 31.01% |
| DPO 100 | 1 | 0.3370 | 0.3794 | 34.36% |
| Pretrained | 3 | 0.3980 | 0.3889 | 30.96% |
| DPO 100 | 3 | 0.3441 | 0.3625 | 34.31% |

The run retained all 12 checkpoint/prompt/seed groups: 36,366 generated records/18,183 pairs, no failures, 72,276 likelihood targets covered exactly once in 3,126 windows. Unweighted test loss was 1.093343127 → 1.084189513. Mean one-road outputs grew 12.084 → 13.503 roads versus 22.143 references; 3,290.2 → 3,555.5 m versus 7,545.0 m. Real cleaned-section revisit rates were only 3.23%/3.34% for the two prompt-eligible populations.

## Notebook-compatible raw-reference primary rows

| Metric, prompt 1, seed 42 | Pretrained | DPO 100 |
| --- | ---: | ---: |
| Trip-distance count-mixing divergence | 0.233230 | 0.212182 |
| Radius count-mixing divergence | 0.200423 | 0.187795 |
| Geographic count-mixing divergence | 0.316749 | 0.319106 |
| OD count-mixing divergence | 0.647332 | 0.647560 |
| Off-network rate | Unavailable | Unavailable |

All 11 network-independent metrics were calculated in the historical metric report. The canonical network was not supplied, so its assessment is explicitly incomplete, with null off-network scores. Prepared geography cannot replace it.

Notebook-compatible metrics use complete raw parents and raw observation repetitions. Their count-mixing function mixes unnormalized counts before KL calculations; it is not standard equal-weight JSD and differs from the square-root base-2 cleaned-section paired distances above. Pairwise, privacy and sampling definitions remain in [HISTORY.md](HISTORY.md#raw-reference-notebook-compatible-metrics), with packaged notebook provenance and parity tests.

## Interpretation and evidence

Some distributions improve while revisiting increases; geographic/OD notebook quantities slightly worsen. Longer routes may loop. This is not a validated high-fidelity trip generator or a privacy guarantee.

Only 2,903/3,584 raw test parents contribute retained test sections; three-road prompts cover 2,938 sections. Exact routes overlap training (570 sections/522 parents), raw repetitions affect weighting, multiple sections can share parents, prompts condition outputs, internal validation entered preference generation and the test split had previously been inspected. Parent separation alone does not imply novel routes or independent evidence.

The separately preserved recovery bundle is `../MobilityGPT-results/historical_bundle_20261021_source_only/`: canonical run and all groups under `runs/nyc_pipeline_full_20261020/`; notebook metrics under `evaluation_outputs/mobilitygpt_canonical_incomplete_20261021/`; compact reports and milestone/prompt/seed evidence under `handoff/`. These are historical evidence locations, not defaults, prerequisites or inputs to a fresh run. This revision leaves them untouched. Fresh run results are self-contained in `output/<run-id>/`.
