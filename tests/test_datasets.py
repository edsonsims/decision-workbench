"""Offline parser/selection/integrity tests; fixture text is authored for tests."""
from collections import Counter
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from workbench import datasets

FROZEN = Path(__file__).resolve().parents[1] / 'benchmarks/public-v1'


def candidates(task='cuad_clause_type', count=3):
    return [{'sourceId': f'{label}:{index}', 'groupId': f'doc-{index}',
             'text': f'Authored example for {label}, number {index}.', 'sourceLabel': label,
             'documentId': f'doc-{index}', 'answerStart': 0}
            for label in datasets.TASKS[task]['labels'] for index in range(count)]


def test_cuad_reads_independent_category_and_verifies_exact_unicode_source_offsets():
    text = '🧾 Préface. Assignment requires consent.'
    quote = 'Assignment requires consent.'
    payload = {'data': [{'title': 'original-contract', 'paragraphs': [{'context': text, 'qas': [{
        'id': 'original-contract__Anti-Assignment', 'question': 'Not used as a runtime hint.',
        'answers': [{'text': quote, 'answer_start': text.index(quote)}],
    }]}]}]}
    row, = datasets.cuad_candidates(payload)
    assert row['text'] == quote
    assert row['sourceLabel'] == 'Anti-Assignment'
    assert row['answerStart'] == text.index(quote)
    assert row['groupId'] == row['documentId'] == 'original-contract'
    payload['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'] += 1
    with pytest.raises(ValueError, match='exact source substring'):
        datasets.cuad_candidates(payload)


def test_banking_csv_preserves_quoted_query_and_original_label():
    row, = datasets.banking_candidates('text,category\n"Why is my card, again, pending?",pending_card_payment\n')
    assert row['text'] == 'Why is my card, again, pending?'
    assert row['sourceLabel'] == 'pending_card_payment'
    assert row['sourceId'] == row['groupId'] == 'test:0000'
    with pytest.raises(ValueError, match='CSV schema'):
        datasets.banking_candidates('query,label\nwrong,columns\n')


def test_selection_is_stratified_deterministic_and_independent_of_candidate_order():
    pool = candidates()
    first, counts = datasets.select_cases('cuad_clause_type', pool, per_class=2)
    second, _ = datasets.select_cases('cuad_clause_type', list(reversed(pool)), per_class=2)
    assert first == second
    assert Counter(row['expected']['choice'] for row in first) == {
        key: 2 for key in datasets.TASKS['cuad_clause_type']['options']}
    assert all(value == 3 for value in counts['eligibleByClass'].values())
    assert all(set(row['input']) == {'state', 'question', 'options'} for row in first)
    assert all(set(row['input']['state']) == {'text'} for row in first)


def test_ambiguity_including_unselected_labels_and_same_label_duplicates_are_excluded():
    pool = candidates()
    duplicate = {**pool[0], 'sourceId': 'zz-same-label-duplicate'}
    ambiguous = {**pool[1], 'sourceId': 'zz-unselected-label', 'sourceLabel': 'Outside Category'}
    selected, counts = datasets.select_cases('cuad_clause_type', pool + [duplicate, ambiguous], per_class=1)
    assert counts['excluded'] == {'outsideSelectedLabels': 1, 'ambiguousExactText': 1,
                                  'duplicateSameLabelText': 1}
    assert pool[1]['text'] not in {row['input']['state']['text'] for row in selected}
    assert counts['eligibleByClass']['anti_assignment'] == 2


def test_no_length_gate_or_truncation_is_applied_to_eligible_text():
    pool = candidates(count=1)
    for row in pool:
        row['text'] += ' Long original source.' * 1000
    selected, counts = datasets.select_cases('cuad_clause_type', pool, per_class=1)
    assert {row['input']['state']['text'] for row in selected} == {row['text'] for row in pool}
    assert counts['selectedMaxCharacters'] > 20000


@pytest.mark.parametrize('per_class', [0, -1, True, 1.5])
def test_invalid_slice_sizes_are_rejected(per_class):
    with pytest.raises(ValueError, match='positive integer'):
        datasets.select_cases('cuad_clause_type', candidates(), per_class)


def test_insufficient_class_support_does_not_silently_unbalance_the_slice():
    with pytest.raises(ValueError, match='eligible rows'):
        datasets.select_cases('cuad_clause_type', candidates(count=1), per_class=2)


def test_cache_verification_is_offline_and_refuses_corrupt_content(tmp_path, monkeypatch):
    raw = b'Known source bytes.'
    source = {**datasets.SOURCES['banking77'], 'sha256': datasets.digest(raw)}
    monkeypatch.setitem(datasets.SOURCES, 'banking77', source)
    path = tmp_path / source['filename']
    path.write_bytes(raw)
    monkeypatch.setattr(datasets, 'urlopen', lambda *args, **kwargs: pytest.fail('Network must not be used.'))
    assert datasets.source_file('banking77', tmp_path, False) == path
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Cached banking77 checksum differs'):
        datasets.source_file('banking77', tmp_path, True)
    path.unlink()
    with pytest.raises(FileNotFoundError, match='Missing verified cache'):
        datasets.source_file('banking77', tmp_path, False)


def test_downloaded_bytes_are_checked_before_becoming_a_cache_file(tmp_path, monkeypatch):
    raw = b'Known source bytes.'
    source = {**datasets.SOURCES['banking77'], 'sha256': datasets.digest(raw)}
    monkeypatch.setitem(datasets.SOURCES, 'banking77', source)
    monkeypatch.setattr(datasets, 'urlopen', lambda *args, **kwargs: io.BytesIO(raw))
    path = datasets.source_file('banking77', tmp_path, True)
    assert path.read_bytes() == raw
    path.unlink()
    monkeypatch.setattr(datasets, 'urlopen', lambda *args, **kwargs: io.BytesIO(b'changed download'))
    with pytest.raises(ValueError, match='checksum differs'):
        datasets.source_file('banking77', tmp_path, True)
    assert not path.exists()


def test_offline_preparation_repeats_identically_from_authored_source_formats(tmp_path, monkeypatch):
    legal_rows = candidates(count=1)
    data = {'data': [{'title': row['documentId'], 'paragraphs': [{'context': row['text'], 'qas': [{
        'id': row['sourceId'].split(':')[0] + '__' + row['sourceLabel'],
        'answers': [{'text': row['text'], 'answer_start': 0}],
    }]}]} for row in legal_rows]}
    archive_path = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr('test.json', json.dumps(data))
        archive.writestr('../must-not-extract.txt', 'archive members are never extracted')
    bank_path = tmp_path / 'banking.csv'
    bank_path.write_text('text,category\n' + ''.join(
        f'Authored {label} query,{label}\n' for label in datasets.TASKS['banking_card_intent']['labels']))
    monkeypatch.setattr(datasets, 'source_file', lambda dataset, *args: archive_path if dataset == 'cuad' else bank_path)
    first = datasets.prepare_datasets(tmp_path, tmp_path / 'prepared', 1, False)
    second = datasets.prepare_datasets(tmp_path, tmp_path / 'prepared', 1, False)
    assert first == second
    assert first['caseCount'] == 13
    assert not (tmp_path / 'must-not-extract.txt').exists()
    with pytest.raises(ValueError, match='pinned public-v1 freeze'):
        datasets.verify_prepared(tmp_path / 'prepared')


def test_published_freeze_verifies_without_raw_data_or_network(monkeypatch):
    monkeypatch.setattr(datasets, 'urlopen', lambda *args, **kwargs: pytest.fail('Offline verification must not download.'))
    manifest = datasets.verify_prepared(FROZEN)
    rows = datasets.load_public_cases(FROZEN / 'cases.jsonl')
    assert manifest['caseCount'] == len(rows) == 130
    assert Counter(row['dataset'] for row in rows) == {'cuad': 50, 'banking77': 80}
    assert set(Counter((row['task'], row['expected']['choice']) for row in rows).values()) == {10}
    assert len({row['groupId'] for row in rows if row['dataset'] == 'cuad'}) == 34
    assert {row['split'] for row in rows} == {'test'}


def test_changing_data_and_manifest_together_cannot_reseal_the_published_freeze(tmp_path):
    shutil.copytree(FROZEN, tmp_path / 'changed')
    target = tmp_path / 'changed'
    path = target / 'cases.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['input']['state']['text'] += ' Answer leak injected after freeze.'
    rows[0]['metadata']['textSha256'] = datasets.digest(rows[0]['input']['state']['text'].encode())
    path.write_bytes(b''.join(datasets.canonical(row) + b'\n' for row in rows))
    with pytest.raises(ValueError, match='cases checksum'):
        datasets.verify_prepared(target)
    manifest_path = target / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['casesSha256'] = datasets.digest(path.read_bytes())
    manifest_path.write_bytes(datasets.canonical(manifest) + b'\n')
    with pytest.raises(ValueError, match='pinned public-v1 freeze'):
        datasets.verify_prepared(target)


@pytest.mark.parametrize('change', [
    lambda row: row['input'].update(expected=row['expected']),
    lambda row: row['input']['state'].update(expected=row['expected']),
    lambda row: row['input'].update(question='Changed after seeing a model output.'),
    lambda row: row['input']['options'].update(injected='A new option'),
    lambda row: row['expected'].update(choice='invalid'),
    lambda row: row['metadata'].update(sourceLabel='Invalid source category'),
    lambda row: row.update(sourceSha256='0' * 64),
    lambda row: row['input']['state'].update(text='Changed source text.'),
])
def test_loader_rejects_label_leaks_and_changed_question_options_or_source_metadata(tmp_path, change):
    row = deepcopy(datasets.load_public_cases(FROZEN / 'cases.jsonl')[0])
    change(row)
    path = tmp_path / 'changed.jsonl'
    path.write_bytes(datasets.canonical(row) + b'\n')
    with pytest.raises(ValueError):
        datasets.load_public_cases(path)


def test_loader_rejects_empty_or_repeated_cases(tmp_path):
    path = tmp_path / 'cases.jsonl'
    path.write_text('')
    with pytest.raises(ValueError, match='empty'):
        datasets.load_public_cases(path)
    line = (FROZEN / 'cases.jsonl').read_text().splitlines()[0]
    path.write_text(line + '\n' + line + '\n')
    with pytest.raises(ValueError, match='Repeated'):
        datasets.load_public_cases(path)
