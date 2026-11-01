"""Contract tests and public-data integrity checks; no neural predictions."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from workbench import banking_v2 as v2
from workbench.datasets import canonical, digest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'benchmarks/banking-public-v2'
V1 = ROOT / 'benchmarks/public-v1'


def pools():
    return {split: [{'sourceId': f'{split}:{i:04d}', 'text': f'{split} example {i} {label}',
                     'label': label} for i, label in enumerate(v2.SPEC['options'])]
            for split in ['train', 'test']}


def selection(rows=None, old=None):
    rows = rows or pools()
    return v2.select(rows['train'], rows['test'], old or [], per_class=1)


def test_published_freeze_and_original_v1_verify():
    manifest = v2.verify(DATA, V1)
    assert manifest['inferencePerformed'] is False
    assert manifest['priorV1CasesSha256'] == '07de590102887155a3228d4662899403d393e7e2b65eb060c098bcee21eba552'
    assert digest((DATA / 'manifest.json').read_bytes()) == v2.FROZEN_MANIFEST_SHA256
    assert manifest['counts']['development']['sourceRows'] == 10003
    assert manifest['counts']['test']['sourceRows'] == 3080
    assert all(manifest['counts'][split]['selectedRows'] == 160 for split in v2.SPLITS)


def test_actual_ids_text_and_splits_have_no_exact_overlap():
    rows = {split: [json.loads(line) for line in (DATA / f'{split}.jsonl').read_text().splitlines()]
            for split in v2.SPLITS}
    old = [json.loads(line) for line in (V1 / 'cases.jsonl').read_text().splitlines()]
    assert len({x['id'] for values in rows.values() for x in values}) == 320
    assert not ({x['input']['state']['text'] for x in rows['development']}
                & {x['input']['state']['text'] for x in rows['test']})
    assert not ({x['sourceId'] for x in rows['test']}
                & {x['sourceId'] for x in old if x['dataset'] == 'banking77'})
    assert not ({x['input']['state']['text'] for values in rows.values() for x in values}
                & {x['input']['state']['text'] for x in old})


def test_selection_is_reproducible_and_keeps_unicode_and_full_length():
    rows = pools()
    text = ' é\r\n' + 'literal🙂 ' * 1500
    rows['train'][0]['text'] = text
    first, counts = selection(rows)
    second, _ = selection(deepcopy(rows))
    assert first == second
    assert first['development'][0]['input']['state']['text'] == text
    assert first['development'][0]['textSha256'] == digest(text.encode())
    assert counts['development']['selectedRows'] == 8


@pytest.mark.parametrize('other_split', ['train', 'test'])
def test_ambiguity_in_unselected_labels_in_either_split_excludes_selected_text(other_split):
    rows = pools()
    rows[other_split].append({'sourceId': f'{other_split}:9999',
                             'text': rows['test'][0]['text'], 'label': 'unselected_intent'})
    with pytest.raises(ValueError, match='insufficient'):
        selection(rows)


def test_any_official_test_text_is_excluded_from_development_before_test_selection():
    rows = pools()
    # The additional official test row need not be chosen by the test selector.
    rows['test'].append({'sourceId': 'test:9999', 'text': rows['train'][0]['text'],
                         'label': rows['train'][0]['label']})
    with pytest.raises(ValueError, match='development/card_arrival'):
        selection(rows)


def test_duplicate_retains_earliest_numeric_csv_row_not_lexical_sort():
    rows = pools()
    original = rows['train'][0]
    original['sourceId'] = 'train:1001'
    rows['train'].append(dict(original, sourceId='train:10000'))
    result, counts = selection(rows)
    assert result['development'][0]['sourceId'] == 'train:1001'
    assert counts['development']['excluded']['sameLabelDuplicate'] == 1


@pytest.mark.parametrize('mode', ['id', 'text'])
def test_v1_exclusion_uses_source_id_or_exact_text(mode):
    rows = pools()
    row = rows['test'][0]
    old = [{'dataset': 'banking77' if mode == 'id' else 'cuad',
            'sourceId': row['sourceId'] if mode == 'id' else 'some-contract-id',
            'input': {'state': {'text': 'different' if mode == 'id' else row['text']}}}]
    with pytest.raises(ValueError, match='test/card_arrival'):
        selection(rows, old)


@pytest.mark.parametrize('count', [True, False, 0, -1, 1.5])
def test_no_invalid_count_or_silent_count_relaxation(count):
    rows = pools()
    with pytest.raises(ValueError):
        v2.select(rows['train'], rows['test'], [], per_class=count)


@pytest.mark.parametrize('data', [b'text,wrong\nhello,label\n',
                                b'text,category\nhello\n',
                                b'text,category\nhello,label,extra\n', b'\xff'])
def test_malformed_csv_rejected(data):
    with pytest.raises((ValueError, UnicodeDecodeError)):
        v2.read_csv(data, 'train')


def test_csv_exact_newlines_and_quoted_text():
    rows = v2.read_csv(b'text,category\r\n"hello\r\nworld",card_arrival\r\n', 'train')
    assert rows == [{'sourceId': 'train:0000', 'text': 'hello\r\nworld', 'label': 'card_arrival'}]
    with pytest.raises(ValueError):
        v2.read_csv(b'text,category\n', 'dev')


def test_only_runtime_input_four_distinct_rotations_no_labels_or_ids():
    case = selection()[0]['development'][0]
    case.update({'id': 'must-stay-out', 'expected': {'choice': 'must-stay-out'}})
    requests = v2.requests(case)
    assert len(requests) == 4
    original = list(v2.SPEC['options'].items())
    for offset, request in zip(v2.OFFSETS, requests, strict=True):
        assert set(request) == {'state', 'question', 'options'}
        assert list(request['options'].items()) == original[offset:] + original[:offset]
        assert 'must-stay-out' not in json.dumps(request)
    requests[0]['state']['text'] = 'mutated'
    requests[0]['options']['card_arrival'] = 'mutated'
    assert requests[1]['state']['text'] == case['input']['state']['text']
    assert case['input']['options'] == v2.SPEC['options']


@pytest.mark.parametrize('field', ['expected', 'sourceId', 'caseId'])
def test_injected_runtime_fields_rejected(field):
    case = selection()[0]['development'][0]
    case['input'][field] = 'must-stay-out'
    with pytest.raises(ValueError):
        v2.requests(case)


def test_altered_task_and_labels_rejected():
    case = selection()[0]['development'][0]
    case['input']['options']['unexpected'] = 'unexpected'
    with pytest.raises(ValueError):
        v2.requests(case)


def test_definition_is_independent_from_callers():
    result = v2.definition()
    result['futureMethod']['orders'][0] = 999
    result['options'][0][1] = 'changed'
    assert v2.definition()['futureMethod']['orders'] == [0, 1, 3, 5]
    assert list(v2.SPEC['options'].values())[0] != 'changed'


def test_attempt_to_reseal_edited_data_and_manifest_fails(tmp_path):
    for path in DATA.glob('*.json*'):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    cases = tmp_path / 'test.jsonl'
    rows = [json.loads(line) for line in cases.read_text().splitlines()]
    rows[0]['input']['state']['text'] = 'changed'
    cases.write_bytes(b''.join(canonical(row) + b'\n' for row in rows))
    manifest_path = tmp_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['test.jsonl'] = digest(cases.read_bytes())
    manifest_path.write_bytes(canonical(manifest) + b'\n')
    with pytest.raises(ValueError, match='Manifest differs'):
        v2.verify(tmp_path, V1)


def test_frozen_case_tamper_fails(tmp_path):
    for path in DATA.glob('*.json*'):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    with (tmp_path / 'development.jsonl').open('ab') as stream:
        stream.write(b'\n')
    with pytest.raises(ValueError, match='Frozen cases differ'):
        v2.verify(tmp_path, V1)


def test_prepare_corrupt_raw_cache_fails_before_output_creation(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'banking77-train.csv').write_bytes(b'corrupt')
    target = tmp_path / 'output'
    with pytest.raises(ValueError, match='train source hash mismatch'):
        v2.prepare(cache, V1, target)
    assert not target.exists()
