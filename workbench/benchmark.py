"""Frozen public-data classification, separate from workflow contract fixtures."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from .backends import (Decision, DecisionTuneBackend, MANIFEST_SHA256,
                       MODEL_ID, MODEL_REVISION, checked_decision)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = 'public-classification-v1'
WARMUP_INPUT = {
    'state': {'text': 'A customer asks about a duplicate charge.'},
    'question': 'Which team should handle this request?',
    'options': {'billing': 'Billing and payment support', 'technical': 'Technical support'},
}


def input_sha256(request: dict) -> str:
    """Preserve option insertion order, unlike sorted-key canonical JSON."""
    return hashlib.sha256(json.dumps(request, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _invoke(backend: Any, request: dict) -> dict:
    # Neither gold labels, task IDs, annotation categories nor source metadata
    # are passed to a provider. A fresh copy also isolates the order probe.
    supplied = deepcopy(request)
    started = time.perf_counter()
    try:
        answer = backend.choose(supplied['state'], supplied['question'], supplied['options'])
        if not isinstance(answer, Decision) or answer.backend != backend.name:
            raise ValueError('Provider returned an invalid decision identity.')
        verified = checked_decision(answer.choice, dict(answer.probabilities),
                                    request['options'], backend.name)
        result = {'prediction': verified.choice, 'decision': verified.as_dict(), 'error': None}
    except Exception as exc:
        # Failures are retained in the denominator. Do not leak arbitrary
        # exception messages, credentials or local paths into public artifacts.
        result = {'prediction': None, 'decision': None, 'error': type(exc).__name__}
    result['elapsedSeconds'] = time.perf_counter() - started
    result['inputSha256'] = input_sha256(request)
    result['optionOrder'] = list(request['options'])
    result['inputEncoding'] = deepcopy(getattr(backend, 'last_input_metadata', None))
    return result


def _validate_cases(cases: list[dict]) -> None:
    if not cases:
        raise ValueError('The public benchmark needs at least one case.')
    ids: set[str] = set()
    tasks: dict[str, tuple] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get('id'), str) or not case['id']:
            raise ValueError('Every case needs a nonempty string id.')
        if case['id'] in ids:
            raise ValueError('Duplicate case id.')
        ids.add(case['id'])
        task = case.get('task')
        if not isinstance(task, str) or not task:
            raise ValueError('Every case needs a task.')
        request = case.get('input')
        if not isinstance(request, dict) or set(request) != {'state', 'question', 'options'}:
            raise ValueError('Only state, question and options may enter a provider.')
        if (not isinstance(request['state'], dict) or set(request['state']) != {'text'}
                or not isinstance(request['state']['text'], str) or not request['state']['text'].strip()):
            raise ValueError('Public classification state must contain only text.')
        question, options = request['question'], request['options']
        if (not isinstance(question, str) or not question.strip() or not isinstance(options, dict)
                or not 2 <= len(options) <= 20
                or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in options.items())):
            raise ValueError('Invalid task question or options.')
        expected = case.get('expected')
        if (not isinstance(expected, dict) or set(expected) != {'choice'}
                or not isinstance(expected['choice'], str) or expected['choice'] not in options):
            raise ValueError('Gold must name one supplied option outside input.')
        signature = (question, tuple(options.items()))
        if task in tasks and tasks[task] != signature:
            raise ValueError('A task must use a fixed question and ordered options.')
        tasks[task] = signature
        if not isinstance(case.get('groupId'), str) or not case['groupId']:
            raise ValueError('A stable source group is required for resampling.')
        # Also rejects non-finite or unserializable input before any inference.
        input_sha256(request)


def _metric_rows(records: list[dict], variant: str) -> list[dict]:
    return [{'id': row['id'], 'expected': row['expected'],
             'prediction': row[variant]['prediction'], 'error': row[variant]['error'],
             'elapsedSeconds': row[variant]['elapsedSeconds'], 'group': row['groupId']}
            for row in records]


def summarize(records: list[dict], cases: list[dict], *, bootstrap_samples: int = 2000,
              seed: int = 42) -> dict:
    from .benchmark_metrics import classification_metrics, paired_option_order_metrics

    tasks = {}
    for task in sorted({case['task'] for case in cases}):
        selected = [row for row in records if row['task'] == task]
        definition = next(case for case in cases if case['task'] == task)
        labels = list(definition['input']['options'])
        primary = _metric_rows(selected, 'primary')
        reverse = _metric_rows(selected, 'reversed')
        # CUAD may contribute several clauses from the same document. Unique
        # banking utterances have one group each; both use source-group bootstrap.
        kwargs = {'bootstrap_samples': bootstrap_samples, 'seed': seed, 'cluster_key': 'group'}
        tasks[task] = {
            'labels': labels, 'classCounts': dict(Counter(row['expected'] for row in selected)),
            'sourceGroups': len({row['groupId'] for row in selected}),
            'uniformRandomExpectedAccuracy': 1 / len(labels),
            'primary': classification_metrics(primary, labels, **kwargs),
            'reversed': classification_metrics(reverse, labels, **kwargs),
            'optionOrder': paired_option_order_metrics(primary, reverse, labels, **kwargs),
        }
    attempted = len(records) * 2
    successful = sum(row[variant]['prediction'] is not None for row in records for variant in ('primary', 'reversed'))
    return {
        'protocol': PROTOCOL_VERSION, 'cases': len(records),
        'decisionAttempts': attempted, 'successfulDecisions': successful,
        'failedDecisions': attempted - successful, 'byTask': tasks,
        'timingDefinition': 'Sequential host wall seconds per completed/failed choose call; model loading and one warmup excluded. Not a throughput or cross-device benchmark.',
        'limits': [
            'A fixed stratified slice, not either full official benchmark; report tasks separately.',
            'Source-group percentile bootstrap is descriptive and unstable on small samples, not a model-superiority test.',
            'BANKING77 is a disclosed DecisionTune training source; overlap for other checkpoints is unknown.',
            'No checkpoint is claimed free of pretraining or fine-tuning contamination.',
            'Model option scores are not calibrated confidence and differ in meaning across adapters.',
            'A reverse-order probe measures one perturbation; it is not robustness certification.',
            'Clause-topic classification is not extraction, legal advice or the multi-role review workflow.',
            'Banking support intent classification is not invoice arithmetic or financial decision making.',
        ],
    }


def evaluate_public(cases: list[dict], backend: Any, *, bootstrap_samples: int = 2000,
                    seed: int = 42, on_record=None) -> tuple[dict, list[dict]]:
    """Exactly one primary and one reverse-order call for every frozen case."""
    _validate_cases(cases)
    if type(bootstrap_samples) is not int or not 1 <= bootstrap_samples <= 100_000:
        raise ValueError('bootstrap_samples must be an integer from 1 to 100000.')
    if type(seed) is not int:
        raise ValueError('seed must be an integer.')
    records = []
    for case in cases:
        primary = _invoke(backend, case['input'])
        reversed_request = deepcopy(case['input'])
        reversed_request['options'] = dict(reversed(list(reversed_request['options'].items())))
        perturbed = _invoke(backend, reversed_request)
        record = {key: case[key] for key in ('id', 'task', 'dataset', 'sourceId', 'groupId') if key in case}
        record.update({'expected': case['expected']['choice'], 'primary': primary, 'reversed': perturbed})
        records.append(record)
        if on_record is not None:
            on_record(record)
    return summarize(records, cases, bootstrap_samples=bootstrap_samples, seed=seed), records


def _decisiontune_metadata(backend: DecisionTuneBackend) -> dict:
    return {'neural': True, 'model': MODEL_ID, 'revision': MODEL_REVISION,
            'license': 'Apache-2.0', 'engine': backend.runtime_backend,
            'manifestSha256': MANIFEST_SHA256, 'scoreMeaning': 'Uncalibrated decision-head option softmax',
            'decisionTunePackage': version('decision-tune'), 'maxInputTokens': 8192}


def _source_provenance() -> dict:
    packages = {}
    for name in ('langgraph', 'decision-tune', 'laya-mlx', 'mlx', 'torch', 'transformers',
                 'tokenizers', 'huggingface-hub', 'numpy', 'safetensors'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        'gitHeadAtExecution': commit, 'workingTreeDirtyAtExecution': dirty,
        'sourceFilesSha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((ROOT / 'workbench').glob('*.py'))},
        'lockFilesSha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('requirements*.lock'))},
        'python': platform.python_version(), 'system': platform.system(),
        'machine': platform.machine(), 'packages': packages,
    }


def main() -> None:
    from .datasets import load_public_cases, verify_prepared

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'benchmarks/public-v1')
    parser.add_argument('--backend', required=True, choices=['decisiontune', 'laya-multilingual', 'qwen3-0.6b'])
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if type(args.bootstrap_samples) is not int or not 1 <= args.bootstrap_samples <= 100_000:
        parser.error('--bootstrap-samples must be from 1 to 100000')
    # Verify all artifacts and runtime cases before loading expensive weights.
    manifest = verify_prepared(args.data)
    cases_path = args.data / 'cases.jsonl'
    cases = load_public_cases(cases_path)
    _validate_cases(cases)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be empty: preserve earlier runs rather than overwrite them')
    args.out.mkdir(parents=True, exist_ok=True)
    run_path = args.out / 'run.json'
    try:
        with run_path.open('x') as reservation:
            json.dump({'status': 'reserved', 'backendIdentifier': args.backend}, reservation)
    except FileExistsError:
        parser.error('--out was reserved by another run; use a fresh directory')
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    if args.backend == 'decisiontune':
        backend = DecisionTuneBackend(args.model_path, 'mlx')
        model_metadata = _decisiontune_metadata(backend)
    else:
        from .comparison_models import create_comparison_backend
        backend = create_comparison_backend(args.backend, args.model_path)
        model_metadata = backend.metadata()
    load_seconds = time.perf_counter() - started
    warmup = _invoke(backend, WARMUP_INPUT)
    provenance = {
        **_source_provenance(), 'model': model_metadata, 'startedAtUtc': started_at,
        'dataManifestSha256': hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
        'casesSha256': hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        'dataManifest': manifest, 'bootstrapSamples': args.bootstrap_samples, 'seed': args.seed,
        'primaryThenReversePerCase': True, 'modelLoadSeconds': load_seconds,
        'warmup': {'input': WARMUP_INPUT, 'result': warmup, 'excludedFromBenchmark': True},
    }
    run_path.write_text(json.dumps({'status': 'running', 'backend': backend.name, 'provenance': provenance}, indent=2) + '\n')
    with (args.out / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        def persist(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps({'completedCases': persist.count + 1, 'totalCases': len(cases),
                              'id': row['id'], 'failures': int(row['primary']['error'] is not None)
                              + int(row['reversed']['error'] is not None)}), flush=True)
            persist.count += 1
        persist.count = 0
        summary, _ = evaluate_public(cases, backend, bootstrap_samples=args.bootstrap_samples,
                                     seed=args.seed, on_record=persist)
    summary['backend'] = backend.name
    summary['provenance'] = provenance
    summary['completedAtUtc'] = datetime.now(timezone.utc).isoformat()
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    run_path.write_text(json.dumps({'status': 'complete', 'backend': backend.name,
                                   'cases': len(cases), 'provenance': provenance}, indent=2) + '\n')
    print(json.dumps({'backend': backend.name, 'cases': summary['cases'],
                      'failedDecisions': summary['failedDecisions'],
                      'summary': str(args.out / 'summary.json')}, indent=2))
    if summary['failedDecisions']:
        parser.exit(1, 'Completed with provider failures; retained in metrics and predictions.\n')


if __name__ == '__main__':
    main()
