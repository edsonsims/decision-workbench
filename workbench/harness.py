"""Evaluate authored development cases; expected labels never enter the workflow."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import time

# This workbench writes local artifacts, even if the invoking shell enabled
# hosted LangSmith tracing for an unrelated project.
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from .backends import (MODEL_ID, MODEL_REVISION, MANIFEST_SHA256,
                       DecisionTuneBackend, ReorderedBackend, RuleBackend)

ROOT = Path(__file__).resolve().parents[1]


def load_cases(path: Path) -> list[dict]:
    cases, ids = [], set()
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) - {'input', 'expected', 'note'} or not {'input', 'expected'} <= set(row):
            raise ValueError('Each case needs input and expected, with an optional note.')
        case_id = row['input']['id']
        if case_id in ids:
            raise ValueError(f'Duplicate case id: {case_id}')
        ids.add(case_id)
        cases.append(row)
    if not cases:
        raise ValueError('The evaluation dataset is empty.')
    return cases


def evaluate(cases: list[dict], backend, min_margin: float = 0.15,
             probe_order: bool = True, routing: str = 'shape') -> tuple[dict, list[dict]]:
    from .workflow import run_workflow

    if not math.isfinite(min_margin) or not 0 <= min_margin <= 1:
        raise ValueError('min_margin must be finite and between zero and one.')
    records = []
    for case in cases:
        # A JSON round trip isolates each run and excludes the expected answer.
        task = json.loads(json.dumps(case['input']))
        started = time.perf_counter()
        result = run_workflow(task, backend, min_margin=min_margin, max_model_calls=3, routing=routing)
        elapsed = time.perf_counter() - started
        expected = case['expected']
        analysis = result.get('analysis', {})
        if not isinstance(analysis, dict):
            analysis = {}
        trace = result.get('trace', [])
        decisions = [row for row in trace if 'decision' in row]
        contract_errors = []
        if result.get('modelCalls', 0) > 3 or len(decisions) > 3:
            contract_errors.append('decision_budget_exceeded')
        if result.get('route') == 'legal' and not result.get('requiresHumanReview'):
            contract_errors.append('legal_review_gate_missing')
        if result.get('route') in {'legal', 'finance'} and analysis and not analysis.get('sourceVerified'):
            if not result.get('requiresHumanReview'):
                contract_errors.append('unverified_evidence_without_review')

        record = {'id': task['id'], 'expected': expected, 'result': result,
                  'routeCorrect': result.get('route') == expected['route'],
                  'categoryCorrect': (analysis.get('category') == expected['category']
                                      if 'category' in expected else None),
                  'contractErrors': contract_errors, 'elapsedSeconds': elapsed}
        if probe_order:
            reordered = run_workflow(json.loads(json.dumps(task)), ReorderedBackend(backend),
                                     min_margin=min_margin, max_model_calls=3, routing=routing)
            first = {row['role']: row['decision']['choice'] for row in decisions}
            second = {row['role']: row['decision']['choice']
                      for row in reordered.get('trace', []) if 'decision' in row}
            record['optionOrderProbe'] = {
                'decisionCalls': reordered.get('modelCalls', 0),
                'routeChanged': result.get('route') != reordered.get('route'),
                'categoryChanged': analysis.get('category') != (reordered.get('analysis') or {}).get('category'),
                'changedRoles': [role for role in first if role in second and first[role] != second[role]],
                'reorderedResult': reordered,
            }
        records.append(record)

    with_category = [row for row in records if row['categoryCorrect'] is not None]
    by_domain = {}
    for domain in sorted({row['expected']['route'] for row in records}):
        selected = [row for row in records if row['expected']['route'] == domain]
        categories = [row for row in selected if row['categoryCorrect'] is not None]
        by_domain[domain] = {
            'cases': len(selected), 'routeCorrect': sum(row['routeCorrect'] for row in selected),
            'categoryCases': len(categories), 'categoryCorrect': sum(row['categoryCorrect'] for row in categories),
        }
    summary = {
        'backend': backend.name, 'routingMode': routing,
        'datasetKind': 'authored synthetic development fixtures, not held out',
        'cases': len(records), 'routeCorrect': sum(row['routeCorrect'] for row in records),
        'categoryCases': len(with_category), 'categoryCorrect': sum(row['categoryCorrect'] for row in with_category),
        'categoryStageReached': sum(bool((row['result'].get('analysis') or {}).get('decision')) for row in with_category),
        'humanReviewCases': sum(bool(row['result'].get('requiresHumanReview')) for row in records),
        'contractFailures': sum(bool(row['contractErrors']) for row in records),
        'primaryDecisionCalls': sum(row['result'].get('modelCalls', 0) for row in records),
        'orderProbeDecisionCalls': sum(row.get('optionOrderProbe', {}).get('decisionCalls', 0) for row in records),
        'byDomain': by_domain,
        'optionOrderRouteChanges': sum(row.get('optionOrderProbe', {}).get('routeChanged', False) for row in records),
        'optionOrderCategoryChanges': sum(row.get('optionOrderProbe', {}).get('categoryChanged', False) for row in records),
        'failureCases': [row['id'] for row in records if not row['routeCorrect'] or row['categoryCorrect'] is False
                         or row['contractErrors']],
        'limits': ['No legal advice or payment actions',
                   'Finance decisions classify supplied deterministic arithmetic findings',
                   'Neural option scores are uncalibrated; margins are routing heuristics',
                   'Literal citation checks do not establish the semantic correctness of a clause classification',
                   'Fixture results do not establish generalization, commercial performance or model superiority'],
    }
    summary['totalDecisionCalls'] = summary['primaryDecisionCalls'] + summary['orderProbeDecisionCalls']
    summary['primaryStageDecisionCounts'] = dict(Counter(
        entry['role'] for row in records for entry in row['result'].get('trace', []) if 'decision' in entry))
    summary['primaryBackendErrors'] = sum(
        entry.get('error') == 'backend_error' for row in records for entry in row['result'].get('trace', []))
    summary['neuralDecisionAttempts'] = summary['totalDecisionCalls'] if backend.name == 'decisiontune-1.0' else 0
    summary['successfulNeuralDecisions'] = (sum(
        sum('decision' in entry for entry in row['result'].get('trace', []))
        + sum('decision' in entry for entry in row.get('optionOrderProbe', {}).get('reorderedResult', {}).get('trace', []))
        for row in records) if backend.name == 'decisiontune-1.0' else 0)
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT / 'examples/cases.jsonl')
    parser.add_argument('--backend', choices=['rules', 'decisiontune'], default='rules')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--engine', choices=['mlx', 'torch', 'onnx'], default='mlx')
    parser.add_argument('--min-margin', type=float, default=0.15)
    parser.add_argument('--routing', choices=['shape', 'model'], default='shape')
    parser.add_argument('--out', type=Path, default=ROOT / '.local/evaluation')
    parser.add_argument('--no-order-probe', action='store_true')
    args = parser.parse_args()
    if args.backend == 'decisiontune' and args.model_path is None:
        parser.error('--model-path is required; the harness never downloads weights automatically')
    backend = RuleBackend() if args.backend == 'rules' else DecisionTuneBackend(args.model_path, args.engine)
    summary, records = evaluate(load_cases(args.cases), backend, args.min_margin,
                                not args.no_order_probe, args.routing)
    summary['provenance'] = {
        'datasetSha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        'workflowSha256': hashlib.sha256((ROOT / 'workbench/workflow.py').read_bytes()).hexdigest(),
        'sourceFilesSha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sorted((ROOT / 'workbench').glob('*.py'))},
        'python': platform.python_version(), 'platform': platform.system(),
        'machine': platform.machine(), 'langgraph': version('langgraph'),
    }
    if args.backend == 'decisiontune':
        summary['provenance'].update({'model': MODEL_ID, 'modelRevision': MODEL_REVISION,
                                      'manifestSha256': MANIFEST_SHA256,
                                      'decisionTunePackage': version('decision-tune'), 'engine': args.engine})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.out / 'traces.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in records))
    print(json.dumps(summary, indent=2))
    if summary['contractFailures']:
        parser.exit(1, 'Workflow contract failures are recorded in traces.jsonl.\n')


if __name__ == '__main__':
    main()
