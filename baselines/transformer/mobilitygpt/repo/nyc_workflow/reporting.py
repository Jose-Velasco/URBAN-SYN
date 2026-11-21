"""Consolidated paired and notebook-compatible reports; no model execution."""
import json

from .preflight import MISSING_NETWORK


def read(path):
    return json.loads(path.read_text())


def write_report(run, config, caveats):
    from .html_report import write_html
    from .metrics import table
    paired = read(run/'evaluation/metrics.json')
    canonical = read(run/'canonical_metrics/results.json')
    selection = read(run/'selection.json')
    complete = canonical['complete']
    result = dict(passed=True, pipeline_complete=True, assessment=config['assessment'], configuration=config,
        canonical_assessment=dict(status=canonical['status'], complete=complete,
            limitation=None if complete else MISSING_NETWORK),
        preflight=read(run/'preflight.json'), verification=read(run/'verification.json'),
        pretraining=read(run/'pretrain/results.json'), preferences=read(run/'preferences/dataset_audit.json'),
        dpo=read(run/'training/result.json'), selection=selection, evaluation=paired,
        canonical_metrics=canonical, stages=read(run/'stage_status.json'),
        reporting_convention=dict(selected_update=config['dpo']['selected_update'], prompt_roads=1, seed=42,
            baseline='matching pretrained row', seed_selection='fixed convention, not best-seed selection'),
        artifacts=dict(prepared=str(run/'prepared'), pretrained=str(run/'pretrain/model_best.pt'),
            selected_dpo=str(run/'model_selected.pt'), evaluation=str(run/'evaluation'),
            canonical_metrics=str(run/'canonical_metrics')))
    (run/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (run/'metrics.json').write_text(json.dumps(paired, indent=2, allow_nan=False)+'\n')
    primary = ['| Model, prompt 1, seed 42 | Count JS distance | Meter JS distance | Revisit |',
               '| --- | ---: | ---: | ---: |']
    for label, prompts in paired['per_seed'].items():
        row = prompts['1']['42']
        primary.append(f"| {label} | {row['road_count_js_distance']:.6f} | {row['distance_m_js_distance']:.6f} | "
                       f"{row['repeated_road_route_rate']:.2%} |")
    status = ('Canonical assessment complete with the supplied network.' if complete else MISSING_NETWORK)
    text = (f"# {config['assessment']}\n\nPipeline completed. {status}\n\n"
        f"Selected checkpoint: [model_selected.pt](model_selected.pt), update {selection['selected_update']}; "
        "selection was locked before training/evaluation. [Pretrained baseline](pretrain/model_best.pt).\n\n"
        "[Paired evaluation](evaluation/report.md) · [Notebook-compatible metrics](canonical_metrics/report.md) · "
        "[Canonical JSON](canonical_metrics/results.json) · [Consolidated JSON](report.json) · "
        "[Artifact hashes](artifact_manifest.json) · [Configuration](pipeline_config.json) · "
        "[Stage commands and timing](stage_status.json)\n\n"
        + '\n'.join(primary) + '\n\nPrimary rows use the matching prompt/seed baseline, not a best-seed claim. '
        'All checkpoint/prompt/seed groups remain in the linked reports.\n\n'
        '## Raw-reference notebook-compatible metrics\n\n' + table(canonical['results'])
        + '\n\nCount-mixing divergences here differ from standard JSD and from the cleaned-section paired distances. '
        'Distribution improvements can coexist with increased revisiting. This is not a validated high-fidelity trip generator.\n\n'
        + '\n'.join('- '+item for item in caveats) + '\n')
    with (run/'report.md').open('x') as stream:
        stream.write(text)
    write_html(run, finalizing=True)
