"""Standalone offline HTML presentation of saved results. Standard library only.

No models, scientific evaluators, metric calculations or notebook imports belong here.
"""
import argparse
from html import escape
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


# Presentation metadata only; the saved evaluator outputs remain authoritative.
PAIRED = (
    ('road_count_js_distance', 'Road-count distribution — JS distance', 'unitless', 'Lower: closer distribution'),
    ('distance_m_js_distance', 'Trip-distance distribution — JS distance', 'unitless; meter bins', 'Lower: closer distribution'),
    ('repeated_road_route_rate', 'Routes revisiting a road', '% of routes', 'Less repetition is not proof of realism'),
    ('immediate_stop_rate', 'Stops immediately after the supplied prompt', '% of routes', 'Descriptive; not a quality score'),
    ('length_cap_rate', 'Routes reaching the generation cap', '% of routes', 'Lower: fewer capped outputs'),
    ('road_count.mean', 'Mean generated route length', 'roads', 'Compare with real routes; not maximize'),
    ('distance_m.mean', 'Mean generated trip distance', 'm', 'Compare with real trips; not maximize'),
)
CANONICAL = (
    ('global_geographic_jsd', 'Geographic visit distribution', 'count-mixing divergence; natural logs', 'Lower: closer distribution'),
    ('od_jsd', 'Origin–destination distribution', 'count-mixing divergence; natural logs', 'Lower: closer distribution'),
    ('trip_distance_jsd', 'Trip-distance distribution', 'count-mixing divergence; natural logs', 'Lower: closer distribution'),
    ('radius_jsd', 'Radius distribution', 'count-mixing divergence; natural logs', 'Lower: closer distribution'),
    ('hausdorff_km', 'Hausdorff route distance', 'km', 'Lower: closer to raw parent'),
    ('normalized_dtw_km', 'Normalized dynamic time warping distance', 'km', 'Lower: closer to raw parent'),
    ('edr', 'Normalized edit distance on real sequences', 'unitless', 'Lower: closer to raw parent'),
    ('jaccard', 'Visited-grid overlap (Jaccard)', 'unitless', 'Higher: greater overlap'),
    ('mia_auc', 'Geometric membership-attack AUC', 'unitless', 'Near 0.5: weak measured attack only'),
    ('uniqueness', 'Sampled training-route separation', '% of sampled generated routes', 'Higher: more separation under this test only'),
    ('off_network_rate', 'Off-network rate', '% of representative points', 'Lower: closer to canonical roads'),
    ('rsc', 'Directed road-transition validity', '% of transitions', 'Higher: more graph-valid; not trip fidelity'),
)
PERCENT = {'repeated_road_route_rate', 'immediate_stop_rate', 'length_cap_rate',
           'uniqueness', 'off_network_rate', 'rsc'}
STAGE_NAMES = dict(preflight='Input, dependency and CUDA checks', prepare='NYC preprocessing',
    gravity='Training-only gravity fit', verify='Export and sampling verification', safety='Pretraining safety gate',
    pretrain='Fresh pretraining', preferences='Preference generation', dpo='DPO training',
    select='Configured checkpoint selection', evaluation_audit='Paired evaluation audit',
    evaluate='Paired generation evaluation', canonical_metrics='Notebook-compatible metrics', report='Final reports')

CSS = """
:root{color-scheme:light;--ink:#172a39;--muted:#465968;--line:#ccd8df;--accent:#175b75}
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:28px 24px 64px}header{margin-bottom:24px}
h1{font-size:2.1rem;line-height:1.2}h2{font-size:1.5rem}h3{font-size:1.13rem}h1,h2,h3{color:#153b50}
section{margin:22px 0;padding:24px;background:white;border:1px solid var(--line);border-radius:10px}
a{color:var(--accent);text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:3px solid #e69e26}
nav{display:flex;flex-wrap:wrap;gap:8px 22px}.muted,caption{color:var(--muted)}.status{font-size:1.4rem;font-weight:650}
.notice{background:#fff7df;border-left:4px solid #b48622;padding:12px 16px}.facts{display:flex;flex-wrap:wrap;gap:16px 40px}
.facts dt{color:var(--muted);font-size:.9rem}.facts dd{margin:0;font-weight:600}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.91rem;margin:12px 0 24px}caption{text-align:left;padding:8px 0}
th,td{border-bottom:1px solid var(--line);padding:10px 12px;text-align:left;vertical-align:top}thead{background:#eaf1f5}
td.number{font-variant-numeric:tabular-nums;white-space:nowrap}td.units,td.direction{font-size:.84rem;color:var(--muted)}
details{border-top:1px solid var(--line);padding:14px 0}summary{cursor:pointer;font-weight:650;color:#153b50}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:16px;font-size:.83rem}code{overflow-wrap:anywhere}
.stages{columns:2}.file-guide li{margin-bottom:8px}.unavailable{color:#68571f}small{font-size:.85rem}
@media(max-width:700px){main{padding:16px 12px 36px}section{padding:16px}.stages{columns:1}h1{font-size:1.7rem}}
"""


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate saved JSON key {key!r}: {path}')
            result[key] = value
        return result
    with path.open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=unique)


def load_saved(run, finalizing=False):
    """Read only run-local saved results; never search repository history."""
    notes, sources, cache = [], {}, {}
    def consolidated():
        if 'report' not in cache:
            cache['report'] = read_json(run/'report.json') if (run/'report.json').is_file() else {}
        return cache['report']
    def record(name, field, default):
        if (run/name).is_file():
            sources[field] = name
            return read_json(run/name)
        fallback = consolidated().get(field)
        if fallback is not None:
            notes.append(f'{name} is absent; using its saved {field} record in report.json.')
            sources[field] = 'report.json'
            return fallback
        notes.append(f'{name} is absent; {field} is unavailable, not recalculated.')
        return default
    config = record('pipeline_config.json', 'configuration', {})
    if not config:
        raise ValueError('Saved run configuration is required (pipeline_config.json or report.json).')
    # The final journal is authoritative, not the report's embedded construction-time snapshot.
    if (run/'stage_status.json').is_file():
        stages = read_json(run/'stage_status.json')
        if not stages or not all(isinstance(v, dict) for v in stages.values()):
            raise ValueError('Invalid/empty saved stage journal')
        unfinished = {name:r.get('status') for name,r in stages.items() if r.get('status') != 'complete'}
        if unfinished and not (finalizing and unfinished == {'report': 'running'}):
            raise ValueError(f'Run is not completed; unfinished stages: {unfinished}')
    elif consolidated().get('pipeline_complete') is True or consolidated().get('passed') is True:
        stages = {}
        notes.append('Final stage journal is absent; completion is reported by saved report.json, not newly verified.')
    else:
        raise ValueError('Run completion cannot be confirmed from a stage journal or completed saved report.')
    selection = record('selection.json', 'selection', {})
    update = selection.get('selected_update', config.get('dpo', {}).get('selected_update'))
    configured = config.get('dpo', {}).get('selected_update')
    if configured is not None and update != configured:
        raise ValueError('Saved selection and configured DPO update disagree')
    if update is None:
        notes.append('Selected DPO update is not recorded; it is not inferred from a filename.')
    paired = record('evaluation/metrics.json', 'evaluation', {})
    canonical = record('canonical_metrics/results.json', 'canonical_metrics', {})
    rows = canonical.get('results', [])
    if not isinstance(rows, list):
        raise ValueError('Invalid saved canonical metric rows')
    canonical_groups = {}
    for row in rows:
        key = (row['checkpoint'], row['prompt_length'], row['seed'])
        if key in canonical_groups:
            raise ValueError(f'Duplicate canonical metric group: {key}')
        canonical_groups[key] = row
    return dict(run=run, config=config, stages=stages, selection=selection, update=update,
        selected=f'selected_dpo_{update}' if update is not None else 'selected_dpo',
        paired=paired, canonical=canonical, canonical_groups=canonical_groups,
        notes=notes, sources=sources, finalizing=finalizing)


def file_link(run, relative, label=None):
    """Link only existing run-owned artifacts; neither recorded absolute paths nor URLs are trusted."""
    relative = Path(relative)
    label = str(label if label is not None else relative)
    target = run/relative
    if relative.is_absolute() or not target.resolve().is_relative_to(run) or not target.exists():
        return escape(label)+' <span class="unavailable">(not available locally)</span>'
    return f'<a href="{escape(quote(relative.as_posix(), safe="/"), quote=True)}">{escape(label)}</a>'


def value_at(row, key):
    for part in key.split('.'):
        if not isinstance(row, dict):
            return None
        row = row.get(part)
    return row


def value_cell(value, key, checkpoint, prompt, seed, reason=None):
    attributes = (f'data-metric="{escape(key, quote=True)}" data-checkpoint="{escape(str(checkpoint), quote=True)}" '
                  f'data-prompt="{escape(str(prompt), quote=True)}" data-seed="{escape(str(seed), quote=True)}"')
    if not numeric(value):
        reason = reason or 'Not recorded or nonfinite in saved results; not recalculated.'
        return f'<td class="unavailable" {attributes} data-value="null">Unavailable <small>— {escape(str(reason))}</small></td>'
    exact = json.dumps(value, allow_nan=False)
    display = f'{value*100:.6g}%' if key in PERCENT else f'{value:.6g}'
    return (f'<td class="number" {attributes} data-value="{escape(exact, quote=True)}" '
            f'title="Saved value: {escape(exact, quote=True)}">{display}</td>')


def comparison_table(data, specs, kind, prompt=1, seed=42):
    labels = ('pretrained', data['selected'])
    if kind == 'paired':
        rows = [data['paired'].get('per_seed', {}).get(label, {}).get(str(prompt), {}).get(str(seed), {}) for label in labels]
        caption = 'Paired, cleaned test sections; prompts and revisits are included.'
    else:
        rows = [data['canonical_groups'].get((label, prompt, seed), {}) for label in labels]
        caption = 'Notebook-compatible metrics against raw NYC observations and complete raw parent trips.'
    body = []
    for key, title, units, direction in specs:
        cells = ''.join(value_cell(value_at(row, key), key, label, prompt, seed,
            row.get('unavailable_reasons', {}).get(key)) for label,row in zip(labels,rows))
        body.append(f'<tr><th scope="row">{escape(title)}</th>{cells}<td class="units">{escape(units)}</td>'
                    f'<td class="direction">{escape(direction)}</td></tr>')
    return (f'<div class="scroll"><table data-kind="{kind}"><caption>{caption}</caption><thead><tr>'
        '<th scope="col">Metric</th><th scope="col">Pretrained</th><th scope="col">Selected DPO</th>'
        '<th scope="col">Units</th><th scope="col">How to read it</th></tr></thead><tbody>'
        + ''.join(body)+'</tbody></table></div>')


def interpretation(data):
    each = data['paired'].get('per_seed', {})
    baseline = each.get('pretrained', {}).get('1', {}).get('42', {})
    selected = each.get(data['selected'], {}).get('1', {}).get('42', {})
    lines = []
    for key, label in (('road_count_js_distance', 'road-count distribution'),
                       ('distance_m_js_distance', 'trip-distance distribution')):
        a, b = baseline.get(key), selected.get(key)
        if numeric(a) and numeric(b):
            relation = 'closer to' if b < a else 'farther from' if b > a else 'equally close to'
            lines.append(f'For the fixed reporting row, DPO’s {label} is {relation} the cleaned reference distribution.')
    a, b = baseline.get('repeated_road_route_rate'), selected.get('repeated_road_route_rate')
    if numeric(a) and numeric(b):
        change = 'increases' if b > a else 'decreases' if b < a else 'is unchanged'
        lines.append(f'The share of routes revisiting a road {change}. Longer or better-matched lengths can still contain loops.')
    raw_a = data['canonical_groups'].get(('pretrained', 1, 42), {})
    raw_b = data['canonical_groups'].get((data['selected'], 1, 42), {})
    for key, label, _, _ in CANONICAL[:4]:
        a, b = raw_a.get(key), raw_b.get(key)
        if numeric(a) and numeric(b):
            relation = 'closer' if b < a else 'farther apart' if b > a else 'unchanged'
            lines.append(f'Raw-reference {label.lower()}: {relation} under the saved count-mixing calculation.')
    if not lines:
        lines.append('The fixed comparison is unavailable in the saved results; no substitute seed or prompt was selected.')
    return '<ul>'+''.join(f'<li>{escape(line)}</li>' for line in lines)+'</ul>'


def group_details(data):
    groups = set(data['canonical_groups'])
    for label, prompts in data['paired'].get('per_seed', {}).items():
        for prompt, seeds in prompts.items():
            groups.update((label, int(prompt), int(seed)) for seed in seeds)
    prompts = sorted({p for _,p,_ in groups})
    seeds = sorted({s for _,_,s in groups})
    result = [f'<p>{len(groups)} saved checkpoint/prompt/seed groups. No seed ranking or aggregate winner is calculated.</p>']
    for prompt in prompts:
        for seed in seeds:
            result.append(f'<details><summary>{escape(str(prompt))}-road prompt · seed {escape(str(seed))}</summary>'
                + comparison_table(data, PAIRED, 'paired', prompt, seed)
                + comparison_table(data, CANONICAL, 'canonical', prompt, seed)+'</details>')
    # Show unknown recorded checkpoint groups explicitly rather than silently dropping them.
    extras = [row for key,row in data['canonical_groups'].items() if key[0] not in ('pretrained', data['selected'])]
    if extras:
        result.append('<h4>Additional saved checkpoint rows</h4><pre>'+escape(json.dumps(extras, indent=2, allow_nan=False))+'</pre>')
    return ''.join(result)


def definitions():
    return """<h3>Two different distribution calculations</h3>
<p><strong>Existing paired JS distance:</strong> square root of base-2 Jensen–Shannon divergence on
independently normalized histograms of cleaned sections. Unitless; lower means closer distributions.
Trip distances use saved road lengths and meter bins, not a meter-valued JS score.</p>
<p><strong>Notebook-compatible JSD:</strong> the saved function mixes raw counts first,
R = (p + q) / sum(p + q), then calculates ½ KL(p / sum(p) || R) + ½ KL(q / sum(q) || R), using natural logs.
This is a count-mixing divergence, not standard equal-weight JSD when totals differ; it is not square-rooted.
Its bounds and values must not be confused with the paired JS distances. Raw repeated observations and complete
raw-parent references also change the populations and weighting.</p>
<h3>Metric conventions</h3><ul>
<li>Geographic/OD counts: fixed 87 × 90 grid, 0.005° cells; OD uses joint first/last in-bounds cells.</li>
<li>Trip distance: consecutive Haversine kilometers; 1,000 bins through 40.71 km plus overflow.</li>
<li>Radius: mean Haversine distance to the arithmetic coordinate mean, not RMS radius; 100 bins through 10.6 km plus overflow.</li>
<li>Hausdorff: symmetric maximum nearest-point Haversine distance. DTW: FastDTW radius 1 cost divided by path length, in km.</li>
<li>Edit distance: strict &lt;100 m match, normalized by maximum sequence length. Jaccard: unique in-bounds visited-grid intersection/union.</li>
<li>Pairwise references: explicitly joined complete raw parents; up to 5,000 sorted pairs, seed 42. Multiple generated sections may share one parent.</li>
<li>Privacy/proximity: up to 2,000 sampled trajectories, seeds 43/44/45, 32 interpolated points; geometric membership AUC uses negative nearest-generated distances.
All generated sections of the queried test parent are excluded from its candidate pool. Separation is strictly &gt;50 m from sampled training trajectories.</li>
<li>Off-network: representative points strictly &gt;30 m from the projected canonical road network. Missing network means unavailable, never zero.</li>
<li>Road-transition validity: original directed relations, accepting consecutive identical IDs; singleton routes contribute no transitions.</li>
<li>Representative coordinates average road geometry endpoints; Earth radius is 6,371 km. Prompts, repeated visits, immediate stops and caps are retained.</li>
</ul>"""


def render(data):
    run, config = data['run'], data['config']
    mode = 'Smoke run — integration diagnostics only' if config.get('smoke') is True else (
        'Full training' if config.get('smoke') is False else 'Run type not recorded')
    canonical_rows = list(data['canonical_groups'].values())
    available = [key for key,_,_,_ in CANONICAL if canonical_rows and all(numeric(row.get(key)) for row in canonical_rows)]
    headline = f'Training completed; {len(available)} of 12 metrics available' if canonical_rows else (
        'Training completed; canonical metric results unavailable')
    unavailable = [name for key,name,_,_ in CANONICAL if key not in available]
    missing = ('<p class="notice">'+escape('Unavailable: '+', '.join(unavailable)+'. ')
        + 'The canonical assessment is not fully available. No missing value is replaced with zero.</p>') if unavailable else ''
    if canonical_rows and 'off_network_rate' not in available:
        reasons = sorted({str(row.get('unavailable_reasons', {}).get('off_network_rate',
            data['canonical'].get('network', {}).get('unavailable_reason', 'Reason not recorded.')))
            for row in canonical_rows if not numeric(row.get('off_network_rate'))})
        missing += '<p>'+escape('Off-network reason: '+'; '.join(reasons))+'. Prepared geometry is not a substitute for the canonical network.</p>'
    completed = [(name,record) for name,record in data['stages'].items() if record.get('status') == 'complete']
    stages = '<ul class="stages">'+''.join('<li>'+escape(STAGE_NAMES.get(name,name))+' — completed</li>' for name,_ in completed)+'</ul>'
    if not data['stages']:
        stages = '<p>Stage-by-stage journal unavailable; saved report records completion.</p>'
    if data['finalizing']:
        stages += '<p class="muted">This page is produced in the final report stage; the linked journal records final artifact sealing.</p>'
    update = str(data['update']) if data['update'] is not None else 'unavailable'
    checkpoint = file_link(run, 'model_selected.pt', 'Selected DPO checkpoint')
    expected = Path('training')/f'model_step_{data["update"]:06d}.pt' if isinstance(data['update'], int) else None
    if not (run/'model_selected.pt').is_file() and expected is not None:
        checkpoint = file_link(run, expected, 'Selected DPO checkpoint')
    target = ''
    if (run/'model_selected.pt').is_symlink():
        target = '<p>Checkpoint symlink target: <code>'+escape(os.readlink(run/'model_selected.pt'))+'</code>.</p>'
    notes = ''.join('<li>'+escape(note)+'</li>' for note in data['notes']) or '<li>Primary saved result files were found locally.</li>'
    files = ('report.md', 'report.json', 'metrics.json', 'evaluation/report.md', 'evaluation/metrics.json',
        'canonical_metrics/report.md', 'canonical_metrics/results.json', 'canonical_metrics/metrics.csv',
        'model_selected.pt', 'pretrain/model_best.pt', 'selection.json', 'prepared', 'preferences',
        'evaluation/generated_trajectories.jsonl', 'logs', 'source', 'pipeline_config.json',
        'environment.json', 'requirements-freeze.txt', 'stage_status.json', 'artifact_manifest.json', 'identity.json')
    links = '<ul>'+''.join('<li>'+file_link(run,name)+'</li>' for name in files)+'</ul>'
    provenance = '<ul>'+''.join('<li>'+escape(key)+': '+file_link(run,name)+'</li>' for key,name in data['sources'].items())+'</ul>'
    verification = data['paired'].get('verification', {})
    extra = '<pre>'+escape(json.dumps(dict(paired_verification=verification,
        canonical_status=data['canonical'].get('status', 'unavailable'),
        canonical_complete=data['canonical'].get('complete'),
        selected_checkpoint=data['selection'], metric_sources=data['sources']), indent=2, allow_nan=False))+'</pre>'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; connect-src 'none'; object-src 'none'">
<title>NYC MobilityGPT results — {escape(run.name)}</title><style>{CSS}</style></head>
<body><main><header><h1>NYC MobilityGPT results</h1><p>Run <code>{escape(run.name)}</code></p>
<p>Most users need only <strong>report.html</strong>. Open this file directly in a browser: no installation, internet or server is needed.</p>
<nav aria-label="Page sections"><a href="#status">Run status</a><a href="#comparison">Main comparison</a>
<a href="#meaning">What the results mean</a><a href="#developer">Developer details</a></nav></header>
<section id="status"><h2>1. Run status</h2><p class="status">{escape(headline)}</p>
<dl class="facts"><div><dt>Execution</dt><dd>{escape(mode)}</dd></div><div><dt>Selected DPO update</dt><dd>{escape(update)}</dd></div>
<div><dt>Metric groups saved</dt><dd>{len(canonical_rows)}</dd></div></dl>{missing}
<p>{checkpoint}; selected update {escape(update)}. {file_link(run,'pretrain/model_best.pt','Matching pretrained baseline')}.</p>{target}
<h3>Completed stages</h3>{stages}</section>
<section id="comparison"><h2>2. Main comparison</h2>
<p><strong>One-road prompt, seed 42</strong>, with the matching pretrained baseline. These are fixed reporting choices, not the best-performing seed.
Other prompts and seeds are supporting evidence in Developer details. No substitute group is chosen if a primary row is absent.</p>
<p class="muted">Display values are rounded for readability; each numeric cell’s tooltip contains its exact saved value (rates are stored as fractions).
All values come from saved results. No metric is recalculated to create this page.</p>
<h3>Paired section comparison</h3>{comparison_table(data,PAIRED,'paired')}
<h3>Notebook-compatible raw-reference metrics</h3>{comparison_table(data,CANONICAL,'canonical')}
<p>The two tables use different populations and distribution formulas. Their JS values are not interchangeable; definitions are expandable below.</p></section>
<section id="meaning"><h2>3. What the results mean</h2>{interpretation(data)}
<p>{'This is a reduced-update smoke test of integration, not full-training quality evidence.' if config.get('smoke') is True else 'These are descriptive observations for this saved run, not guaranteed behavior on other runs or populations.'}</p>
<p>Distribution gains and revisiting tradeoffs must be read together. Matching a marginal distribution does not validate a particular continuation.
No statistical significance or overall winning model is claimed. This is not a validated high-fidelity trip generator.</p>
<p>Coverage excludes some raw test parents; exact routes overlap training; repeated raw observations change weighting; sections may share parents;
supplied prompts condition outputs; preferences reuse internal validation; and the test split was previously inspected.
The 80-road cap cannot reproduce longer reference tails. Privacy metrics describe a sampled geometric attack/proximity test, not a privacy guarantee.</p></section>
<section id="developer"><h2>4. Developer details</h2>
<h3>What are these files?</h3><ul class="file-guide">
<li><strong>Reports and CSVs:</strong> inspect results. Start here in report.html; JSON/CSV retain machine-readable values.</li>
<li><strong>Checkpoints:</strong> load trained models. The selected DPO and matching pretrained files are identified above.</li>
<li><strong>Prepared data and preferences:</strong> intermediate pipeline inputs, needed for auditing or subsequent stages.</li>
<li><strong>Logs, snapshots and manifests:</strong> debugging and reproducibility. Keep them intact; most readers do not need to open them.</li></ul>
<details><summary>Configuration</summary><pre>{escape(json.dumps(config,indent=2,allow_nan=False))}</pre></details>
<details><summary>All prompt/seed groups</summary>{group_details(data)}</details>
<details><summary>Metric definitions, units and interpretation</summary>{definitions()}</details>
<details><summary>Files and local artifact links</summary>{links}</details>
<details><summary>Saved verification and presentation provenance</summary>{provenance}<ul>{notes}</ul>{extra}
<p>This page is a presentation of saved results, not a new scientific evaluation or verification of historical strict-resume identities.
Source changes can invalidate strict resume; restore recorded source/path/environment identities for exact replay. Original manifests are not rewritten.</p></details>
</section></main></body></html>'''


def write_html(run, finalizing=False):
    run = Path(run).expanduser().resolve()
    if not run.is_dir():
        raise ValueError(f'Run directory does not exist: {run}')
    destination = run/'report.html'
    if os.path.lexists(destination):
        raise FileExistsError(f'Refusing to overwrite existing report: {destination}')
    document = render(load_saved(run, finalizing=finalizing))
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(document)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description='Create report.html from saved results only; never overwrite reports or calculate metrics.')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    print(write_html(args.run_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
