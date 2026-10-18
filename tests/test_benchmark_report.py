"""Offline integrity checks; generated predictions are explicit test doubles."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from workbench.backends import checked_decision
from workbench.benchmark import evaluate_public
from workbench import benchmark_report, datasets


FROZEN = Path(__file__).resolve().parents[1] / 'benchmarks/public-v1'


@pytest.fixture(scope='module')
def artifacts():
    manifest = datasets.verify_prepared(FROZEN)
    cases = datasets.load_public_cases(FROZEN / 'cases.jsonl')
    failing_text = cases[0]['input']['state']['text']

    class Backend:
        name = 'offline-report-test-double'

        def choose(self, state, question, options):
            if state['text'] == failing_text:
                raise RuntimeError('Deliberate test failure, not a model inference.')
            first = next(iter(options))
            return checked_decision(first, {key: float(key == first) for key in options}, options, self.name)

    summary, records = evaluate_public(cases, Backend(), bootstrap_samples=20, seed=123)
    provenance = {'dataManifest': manifest, 'casesSha256': manifest['casesSha256'],
                  'dataManifestSha256': hashlib.sha256((FROZEN / 'manifest.json').read_bytes()).hexdigest(),
                  'bootstrapSamples': 20, 'seed': 123}
    summary.update(backend=Backend.name, provenance=provenance)
    return {'summary': summary, 'records': records,
            'run': {'status': 'complete', 'cases': 130, 'backend': Backend.name,
                    'provenance': deepcopy(provenance)}}


def write_run(directory, artifacts):
    directory.mkdir()
    (directory / 'summary.json').write_text(json.dumps(artifacts['summary']))
    (directory / 'run.json').write_text(json.dumps(artifacts['run']))
    (directory / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in artifacts['records']))
    return directory


def test_recompute_all_frozen_cases_offline_including_failed_attempts(tmp_path, artifacts, monkeypatch):
    monkeypatch.setattr(datasets, 'urlopen', lambda *_: pytest.fail('Report verification must remain offline.'))
    directory = write_run(tmp_path / 'run', artifacts)
    verified = benchmark_report.verify_run(directory, FROZEN)
    assert verified == artifacts['summary']
    assert verified['cases'] == 130 and verified['decisionAttempts'] == 260
    assert verified['failedDecisions'] == 2
    clauses = verified['byTask']['cuad_clause_type']['primary']
    assert clauses['cases'] == 50 and clauses['completed'] == 49
    assert clauses['failed'] == 1
    table = benchmark_report.markdown_table([verified])
    assert 'offline-report-test-double' in table and '49/50' in table


@pytest.mark.parametrize('tamper', [
    lambda a: a['records'].pop(),
    lambda a: a['records'].append(deepcopy(a['records'][0])),
    lambda a: a['records'].reverse(),
    lambda a: a['records'][1].update(id=a['records'][0]['id']),
    lambda a: a['records'][0].update(expected='forged-label'),
    lambda a: a['records'][0].update(groupId='forged-document'),
    lambda a: a['records'][0].update(sourceId='forged-source'),
    lambda a: a['records'][0].update(task='forged-task'),
    lambda a: a['records'][1]['primary'].update(inputSha256='0' * 64),
    lambda a: a['records'][1]['reversed']['optionOrder'].reverse(),
    lambda a: a['records'][1]['primary'].update(prediction='forged-label'),
    lambda a: a['records'][1]['primary']['decision'].update(backend='forged-backend'),
    lambda a: a['records'][1]['primary']['decision'].update(margin=.125),
    lambda a: a['records'][1]['primary'].update(elapsedSeconds=float('inf')),
    lambda a: a['records'][1]['primary'].update(error='Failure with a retained valid answer'),
    lambda a: a['records'][0]['primary'].update(error=None),
    lambda a: a['summary']['byTask']['cuad_clause_type']['primary'].update(accuracy=1),
    lambda a: a['summary'].update(failedDecisions=0),
    lambda a: a['summary'].update(cases=129),
    lambda a: a['run'].update(status='running'),
    lambda a: a['run'].update(backend='forged-backend'),
    lambda a: a['run']['provenance'].update(casesSha256='0' * 64),
])
def test_changed_predictions_metrics_or_metadata_cannot_pass_verification(tmp_path, artifacts, tamper):
    changed = deepcopy(artifacts)
    tamper(changed)
    with pytest.raises(ValueError):
        benchmark_report.verify_run(write_run(tmp_path / 'tampered', changed), FROZEN)


def test_resealed_run_metadata_still_cannot_claim_different_frozen_data(tmp_path, artifacts):
    changed = deepcopy(artifacts)
    for artifact in ('run', 'summary'):
        changed[artifact]['provenance']['casesSha256'] = '0' * 64
    with pytest.raises(ValueError, match='different public cases'):
        benchmark_report.verify_run(write_run(tmp_path / 'tampered', changed), FROZEN)


def test_invalid_scores_are_rejected_before_any_table(tmp_path, artifacts):
    changed = deepcopy(artifacts)
    probabilities = changed['records'][1]['primary']['decision']['probabilities']
    probabilities[next(iter(probabilities))] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        benchmark_report.verify_run(write_run(tmp_path / 'invalid-scores', changed), FROZEN)
