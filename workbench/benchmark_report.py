"""Verify frozen public predictions and recompute metrics without model weights."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .backends import checked_decision
from .benchmark import ROOT, input_sha256, summarize
from .datasets import load_public_cases, verify_prepared


def verify_run(directory: Path, data: Path) -> dict:
    manifest = verify_prepared(data)
    cases = load_public_cases(data / 'cases.jsonl')
    run = json.loads((directory / 'run.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    if (run.get('status') != 'complete' or run.get('cases') != len(cases)
            or summary.get('backend') != run.get('backend')
            or summary.get('provenance') != run.get('provenance')):
        raise ValueError('Run is incomplete or metadata disagrees.')
    provenance = summary['provenance']
    if (provenance.get('dataManifest') != manifest
            or provenance.get('casesSha256') != manifest['casesSha256']
            or provenance.get('dataManifestSha256') != hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest()):
        raise ValueError('Run used different public cases or protocol.')
    records = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines() if line]
    if [row.get('id') for row in records] != [case['id'] for case in cases]:
        raise ValueError('Predictions must cover every frozen case exactly once, in order.')
    for case, row in zip(cases, records):
        for key in ('task', 'dataset', 'sourceId', 'groupId'):
            if row.get(key) != case[key]:
                raise ValueError('Prediction source identity differs from the freeze.')
        if row.get('expected') != case['expected']['choice']:
            raise ValueError('Prediction gold label differs from the freeze.')
        for variant in ('primary', 'reversed'):
            request = deepcopy(case['input'])
            if variant == 'reversed':
                request['options'] = dict(reversed(list(request['options'].items())))
            result = row[variant]
            if (result.get('inputSha256') != input_sha256(request)
                    or result.get('optionOrder') != list(request['options'])):
                raise ValueError('Prediction input/order differs from the frozen request.')
            elapsed = result.get('elapsedSeconds')
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Prediction timing must be finite and nonnegative.')
            if result.get('error') is not None:
                if (not isinstance(result['error'], str) or not result['error'].strip()
                        or result.get('prediction') is not None or result.get('decision') is not None):
                    raise ValueError('Provider failures must remain explicit failed predictions.')
            else:
                decision = result.get('decision')
                if not isinstance(decision, dict) or decision.get('backend') != summary['backend']:
                    raise ValueError('Missing or inconsistent decision backend.')
                verified = checked_decision(decision.get('choice'), decision.get('probabilities'),
                                            request['options'], summary['backend'])
                if verified.choice != result.get('prediction') or verified.as_dict() != decision:
                    raise ValueError('Prediction differs from its validated decision.')
    recomputed = summarize(records, cases, bootstrap_samples=provenance['bootstrapSamples'],
                           seed=provenance['seed'])
    if {key: summary.get(key) for key in recomputed} != recomputed:
        raise ValueError('Published metrics differ from predictions.')
    return summary


def markdown_table(summaries: list[dict]) -> str:
    lines = ['| Model | Task | Correct | Macro F1 | Completed | Reversed choices changed | Median call (ms) |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for summary in summaries:
        for task, values in summary['byTask'].items():
            primary, paired = values['primary'], values['optionOrder']
            median = primary['timing']['medianSeconds']
            lines.append(f"| {summary['backend']} | {task} | {primary['correct']}/{primary['cases']} "
                         f"| {primary['macroF1']:.3f} | {primary['completed']}/{primary['cases']} "
                         f"| {paired['choiceChanges']}/{paired['bothCompleted']} | {median * 1000:.1f} |")
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'reference/public-v1')
    parser.add_argument('--data', type=Path, default=ROOT / 'benchmarks/public-v1')
    args = parser.parse_args()
    directories = sorted(path.parent for path in args.runs.glob('*/summary.json'))
    if not directories:
        parser.error('No completed public runs found.')
    summaries = [verify_run(path, args.data) for path in directories]
    print(markdown_table(summaries))
    print(f'\nVerified {len(summaries)} runs; metrics recomputed offline from all frozen predictions.')


if __name__ == '__main__':
    main()
