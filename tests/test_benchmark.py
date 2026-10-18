"""Runner boundary tests use explicit test doubles, never model-quality evidence."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from workbench.backends import Decision, checked_decision
from workbench import benchmark


def cases():
    """130 distinct test rows matching planned counts, not public dataset rows."""
    result = []
    for task, size, classes in [('clauses', 50, 5), ('intents', 80, 8)]:
        options = {f'class-{i}': f'Fixed description for category {i}' for i in range(classes)}
        for i in range(size):
            result.append({'id': f'{task}-{i}', 'task': task, 'dataset': 'test-double-data',
                           'sourceId': f'source-{task}-{i}', 'groupId': f'{task}-document-{i // 2}',
                           'input': {'state': {'text': f'{task} example number {i}'},
                                     'question': 'Choose the text category.', 'options': deepcopy(options)},
                           'expected': {'choice': f'class-{i % classes}'}})
    return result


class RecordingBackend:
    name = 'runner-test-double'

    def __init__(self, fail_text=None, mutate=False):
        self.calls, self.fail_text, self.mutate = [], fail_text, mutate

    def choose(self, state, question, options):
        self.calls.append(deepcopy({'state': state, 'question': question, 'options': options}))
        if state['text'] == self.fail_text:
            raise RuntimeError('Private path /secret/person and token must not enter artifacts')
        answer = checked_decision('class-0', {key: float(key == 'class-0') for key in options}, options, self.name)
        if self.mutate:
            state['text'] = 'changed by backend'
            options.clear()
        return answer


def test_every_planned_id_retained_in_both_orders_including_exceptions():
    planned = cases()
    original = deepcopy(planned)
    backend = RecordingBackend(fail_text='clauses example number 3', mutate=True)
    summary, records = benchmark.evaluate_public(planned, backend, bootstrap_samples=20)
    assert planned == original
    assert len(records) == summary['cases'] == 130
    assert [record['id'] for record in records] == [case['id'] for case in planned]
    assert summary['decisionAttempts'] == len(backend.calls) == 260
    assert summary['failedDecisions'] == 2 and summary['successfulDecisions'] == 258
    failed = next(record for record in records if record['id'] == 'clauses-3')
    for variant in ('primary', 'reversed'):
        assert failed[variant]['prediction'] is None
        assert failed[variant]['error'] == 'RuntimeError'
        assert failed[variant]['elapsedSeconds'] >= 0
    assert 'Private path' not in json.dumps(records)
    assert summary['byTask']['clauses']['primary']['cases'] == 50
    assert summary['byTask']['clauses']['primary']['completed'] == 49
    assert summary['byTask']['intents']['primary']['cases'] == 80
    for i, case in enumerate(original):
        primary, reverse = backend.calls[2 * i:2 * i + 2]
        assert set(primary) == {'state', 'question', 'options'}
        assert primary['state'] == reverse['state'] == case['input']['state']
        assert set(primary['state']) == {'text'}
        assert primary['question'] == reverse['question'] == case['input']['question']
        assert list(primary['options'].items()) == list(reversed(list(reverse['options'].items())))
        assert dict(primary['options']) == dict(reverse['options'])
        assert records[i]['primary']['inputSha256'] != records[i]['reversed']['inputSha256']


@pytest.mark.parametrize('answer', [
    {'choice': 'class-0'},
    Decision('class-0', {'class-0': 1, 'class-1': 0}, 'wrong-backend'),
    Decision('class-0', {'class-0': float('nan'), 'class-1': 0}, 'broken'),
    Decision('outside', {'class-0': 1, 'class-1': 0}, 'broken'),
    Decision('class-0', {'class-0': 1}, 'broken'),
    Decision('class-1', {'class-0': 1, 'class-1': 0}, 'broken'),
    Decision([], {'class-0': 1, 'class-1': 0}, 'broken'),
    Decision('class-0', None, 'broken'),
])
def test_malformed_provider_decisions_become_failed_attempts(answer):
    class Broken:
        name = 'broken'

        def choose(self, *_):
            return answer

    request = {'state': {'text': 'test'}, 'question': 'Class?',
               'options': {'class-0': 'First', 'class-1': 'Second'}}
    result = benchmark._invoke(Broken(), request)
    assert result['prediction'] is None and result['decision'] is None
    assert result['error'] in {'ValueError', 'TypeError'}
    assert result['inputSha256'] == benchmark.input_sha256(request)


@pytest.mark.parametrize('mutation', [
    lambda rows: rows[0]['input']['state'].update(expected='class-0'),
    lambda rows: rows[0]['input'].update(expected='class-0'),
    lambda rows: rows[1]['input']['options'].update({'class-0': 'Changed description'}),
    lambda rows: rows[1]['input'].update(question='Changed question'),
    lambda rows: rows[1].update(id=rows[0]['id']),
    lambda rows: rows[0]['expected'].update(choice='unknown'),
])
def test_invalid_or_label_leaking_cases_rejected_before_any_call(mutation):
    planned, backend = cases(), RecordingBackend()
    mutation(planned)
    with pytest.raises(ValueError):
        benchmark.evaluate_public(planned, backend, bootstrap_samples=20)
    assert backend.calls == []


def test_cli_refuses_existing_artifacts_before_model_loading(tmp_path, monkeypatch):
    planned = cases()[:1]
    data, output = tmp_path / 'data', tmp_path / 'previous-run'
    data.mkdir()
    output.mkdir()
    previous = output / 'predictions.jsonl'
    previous.write_bytes(b'preserve the original evidence\n')
    datasets = SimpleNamespace(verify_prepared=lambda _: {}, load_public_cases=lambda _: planned)
    monkeypatch.setitem(sys.modules, 'workbench.datasets', datasets)
    monkeypatch.setattr(benchmark, 'DecisionTuneBackend', lambda *_: pytest.fail('Model must not load'))
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--data', str(data), '--out', str(output),
                                    '--backend', 'decisiontune', '--model-path', str(tmp_path / 'weights')])
    with pytest.raises(SystemExit) as stopped:
        benchmark.main()
    assert stopped.value.code == 2
    assert previous.read_bytes() == b'preserve the original evidence\n'
    assert list(output.iterdir()) == [previous]


def test_concurrent_cli_runs_cannot_both_claim_same_output(tmp_path, monkeypatch):
    data, output = tmp_path / 'data', tmp_path / 'one-run'
    data.mkdir()
    (data / 'manifest.json').write_text('{}')
    (data / 'cases.jsonl').write_text('{}\n')
    monkeypatch.setitem(sys.modules, 'workbench.datasets', SimpleNamespace(
        verify_prepared=lambda _: {}, load_public_cases=lambda _: cases()[:1]))
    started, release = Event(), Event()
    constructions = []

    class Backend:
        name = 'concurrency-test-double'

        def choose(self, state, question, options):
            choice = next(iter(options))
            return checked_decision(choice, {key: float(key == choice) for key in options}, options, self.name)

    def load(*_):
        constructions.append(True)
        if len(constructions) == 1:
            started.set()
            assert release.wait(timeout=5)
        return Backend()

    monkeypatch.setattr(benchmark, 'DecisionTuneBackend', load)
    monkeypatch.setattr(benchmark, '_decisiontune_metadata', lambda _: {'neural': False})
    monkeypatch.setattr(benchmark, '_source_provenance', lambda: {})
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--data', str(data), '--out', str(output),
                                    '--backend', 'decisiontune', '--model-path', str(tmp_path / 'weights'),
                                    '--bootstrap-samples', '10'])

    def run():
        try:
            benchmark.main()
        except SystemExit as stopped:
            return stopped.code
        return 0

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run)
        assert started.wait(timeout=2)
        try:
            second_status = pool.submit(run).result(timeout=3)
        finally:
            release.set()
        first_status = first.result(timeout=3)
    assert len(constructions) == 1, 'A second run loaded weights and could overwrite the first run.'
    assert sorted([first_status, second_status]) == [0, 2]
    assert len((output / 'predictions.jsonl').read_text().splitlines()) == 1
