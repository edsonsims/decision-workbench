"""Offline preparation and authentication of the separate BANKING77 v2 freeze.

No model loading, prediction reading, threshold fitting or implicit downloads.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path

from .datasets import (BANKING_REVISION, TASKS, canonical, digest,
                       verify_prepared as verify_v1)

PROTOCOL = 'decision-workbench/banking-public-v2'
PER_CLASS = 20
OFFSETS = (0, 1, 3, 5)
FROZEN_MANIFEST_SHA256 = 'c0e109e52fac88a341aef9866cc3b93b0e98e9f928d34eef00e324de7dd756f6'
SOURCE_HASHES = {
    'train': 'b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b',
    'test': 'd12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d',
}
SPEC = TASKS['banking_card_intent']
SPLITS = {'development': 'train', 'test': 'test'}


def definition() -> dict:
    """Return fresh protocol objects; runtime callers cannot mutate task globals."""
    return json.loads(canonical({
        'id': PROTOCOL, 'sourceRevision': BANKING_REVISION,
        'task': 'banking_card_intent', 'perClassPerSplit': PER_CLASS,
        'question': SPEC['question'], 'options': list(SPEC['options'].items()),
        'sourceHashes': SOURCE_HASHES, 'orderOffsets': OFFSETS,
        'selection': 'Lowest SHA256 of canonical [protocol,split,sourceId,textSha256] per class.',
        'exclusions': ['blank text', 'exact text with multiple labels across ALL train/test labels',
                       'duplicate same-label text within split (retain earliest source row)',
                       'all train text occurring anywhere in official test',
                       'all public-v1 banking source IDs and exact text from either task'],
        'normalization': 'None. Exact Unicode text; no length filter or truncation.',
        'limitations': ['Exact separation is not semantic independence or training-unseen evidence.',
                        'BANKING77 is a declared DecisionTune training source; other exposure unknown.',
                        'Eight selected intents are not the full77-intent benchmark.',
                        'No inference, calibration, answer-quality or human-review result is recorded.'],
        'futureMethod': {
            'status': 'planned; not implemented or executed by this data-freeze module',
            'orders': list(OFFSETS), 'thresholdGrid': [f'{x / 20:.2f}' for x in range(21)],
            'selection': 'Development only: at most1/10 accepted cases erroneous; maximize count, then highest threshold. Abstain-all permitted.',
            'gate': 'Require all four valid probes agree; failed/disagreeing cases go to review.',
            'limits': 'An empirical development constraint is not calibrated confidence or a test error guarantee.',
            'test': 'Freeze model/adapter/assets, inference method and development-fitted policy before evaluating test predictions.',
        },
    }))


def read_csv(data: bytes, source_split: str) -> list[dict]:
    """Parse source row IDs without changing source text or inferred labels."""
    if source_split not in SOURCE_HASHES:
        raise ValueError('Unknown official split.')
    reader = csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))
    if reader.fieldnames != ['text', 'category']:
        raise ValueError('Unexpected BANKING77 CSV header.')
    rows = []
    for index, row in enumerate(reader):
        if set(row) != {'text', 'category'} or any(not isinstance(x, str) for x in row.values()):
            raise ValueError('Malformed BANKING77 CSV row.')
        rows.append({'sourceId': f'{source_split}:{index:04d}',
                     'text': row['text'], 'label': row['category']})
    return rows


def select(train: list[dict], test: list[dict], v1: list[dict], *, per_class: int = PER_CLASS) -> tuple[dict, dict]:
    """Hash-rank only after removing ambiguity and known exact overlap.

    The optional count supports small contract fixtures; the published freeze
    always uses the fixed20. No scores or predictions are accepted here.
    """
    if type(per_class) is not int or per_class < 1:
        raise ValueError('per_class must be a positive integer.')
    labels = defaultdict(set)
    for row in train + test:
        labels[row['text']].add(row['label'])
    official_test_texts = {row['text'] for row in test}
    old_texts = {row['input']['state']['text'] for row in v1}
    old_ids = {row['sourceId'] for row in v1 if row['dataset'] == 'banking77'}
    output, counts = {}, {}
    for split, rows in [('development', train), ('test', test)]:
        pools = defaultdict(list)
        excluded = Counter()
        seen = set()
        for row in rows:  # official CSV order: earliest numeric row wins duplicates
            text = row['text']
            if row['label'] not in SPEC['options']:
                excluded['outsideLabels'] += 1
            elif not text.strip():
                excluded['blank'] += 1
            elif len(labels[text]) > 1:
                excluded['crossLabelExactText'] += 1
            elif split == 'development' and text in official_test_texts:
                excluded['inOfficialTest'] += 1
            elif row['sourceId'] in old_ids or text in old_texts:
                excluded['inPublicV1'] += 1
            elif text in seen:
                excluded['sameLabelDuplicate'] += 1
            else:
                seen.add(text)
                pools[row['label']].append(row)
        chosen = []
        for label in SPEC['options']:
            if len(pools[label]) < per_class:
                raise ValueError(f'{split}/{label} has insufficient eligible source rows.')
            def rank(row: dict) -> tuple:
                return (digest(canonical([PROTOCOL, split, row['sourceId'], digest(row['text'].encode('utf-8'))])), row['sourceId'])
            for row in sorted(pools[label], key=rank)[:per_class]:
                chosen.append({
                    'id': f'banking-v2:{split}:{digest(row["sourceId"].encode())[:24]}',
                    'protocol': PROTOCOL, 'split': split, 'sourceSplit': SPLITS[split],
                    'sourceId': row['sourceId'], 'groupId': row['sourceId'],
                    'sourceSha256': SOURCE_HASHES[SPLITS[split]],
                    'input': {'state': {'text': row['text']}, 'question': SPEC['question'],
                              'options': dict(SPEC['options'])},
                    'expected': {'choice': label},
                    'textSha256': digest(row['text'].encode('utf-8')),
                })
        output[split] = chosen
        counts[split] = {'sourceRows': len(rows), 'excluded': dict(sorted(excluded.items())),
                         'eligibleByClass': {label: len(pools[label]) for label in SPEC['options']},
                         'selectedByClass': {label: per_class for label in SPEC['options']},
                         'selectedRows': len(chosen)}
    return output, counts


def prepare(cache: Path, v1_root: Path, out: Path) -> dict:
    """Use only explicitly supplied, hash-verified raw files. Refuse replacement."""
    old_manifest = verify_v1(v1_root)
    v1 = [json.loads(line) for line in (v1_root / 'cases.jsonl').read_text().splitlines()]
    sources = {}
    for source_split, checksum in SOURCE_HASHES.items():
        data = (cache / f'banking77-{source_split}.csv').read_bytes()
        if digest(data) != checksum:
            raise ValueError(f'{source_split} source hash mismatch.')
        sources[source_split] = read_csv(data, source_split)
    selected, counts = select(sources['train'], sources['test'], v1)
    payloads = {f'{split}.jsonl': b''.join(canonical(row) + b'\n' for row in rows)
                for split, rows in selected.items()}
    manifest = {'protocol': definition(), 'counts': counts,
                'priorV1ManifestSha256': digest((v1_root / 'manifest.json').read_bytes()),
                'priorV1CasesSha256': old_manifest['casesSha256'],
                'files': {name: digest(data) for name, data in payloads.items()},
                'inferencePerformed': False}
    payloads['manifest.json'] = canonical(manifest) + b'\n'
    # Validate every existing target before writing anything.
    for name, data in payloads.items():
        path = out / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f'Refusing to replace a different freeze file: {name}')
    out.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        path = out / name
        if not path.exists():
            with path.open('xb') as handle:
                handle.write(data)
    return manifest


def verify(root: Path, v1_root: Path) -> dict:
    """Authenticate the pinned freeze and its runtime/label/split boundary offline."""
    manifest_bytes = (root / 'manifest.json').read_bytes()
    if digest(manifest_bytes) != FROZEN_MANIFEST_SHA256:
        raise ValueError('Manifest differs from the published v2 freeze.')
    manifest = json.loads(manifest_bytes)
    verify_v1(v1_root)
    if manifest['protocol'] != definition() or digest((v1_root / 'manifest.json').read_bytes()) != manifest['priorV1ManifestSha256']:
        raise ValueError('Protocol or original v1 freeze differs.')
    old = [json.loads(line) for line in (v1_root / 'cases.jsonl').read_text().splitlines()]
    old_texts = {row['input']['state']['text'] for row in old}
    old_ids = {row['sourceId'] for row in old if row['dataset'] == 'banking77'}
    texts, ids = set(), set()
    expected_keys = {'id', 'protocol', 'split', 'sourceSplit', 'sourceId', 'groupId',
                     'sourceSha256', 'input', 'expected', 'textSha256'}
    for split, official in SPLITS.items():
        path = root / f'{split}.jsonl'
        data = path.read_bytes()
        if digest(data) != manifest['files'][path.name]:
            raise ValueError('Frozen cases differ.')
        rows = [json.loads(line) for line in data.splitlines()]
        tally = Counter()
        for row in rows:
            runtime = row['input']
            if (set(row) != expected_keys or row['protocol'] != PROTOCOL
                    or row['split'] != split or row['sourceSplit'] != official
                    or not row['sourceId'].startswith(official + ':')
                    or row['sourceSha256'] != SOURCE_HASHES[official]
                    or row['groupId'] != row['sourceId']
                    or set(runtime) != {'state', 'question', 'options'}
                    or set(runtime['state']) != {'text'}
                    or runtime['question'] != SPEC['question']
                    or list(runtime['options'].items()) != list(SPEC['options'].items())
                    or set(row['expected']) != {'choice'}
                    or row['expected']['choice'] not in SPEC['options']):
                raise ValueError('Invalid frozen schema or runtime boundary.')
            text = runtime['state']['text']
            identity = f'banking-v2:{split}:{digest(row["sourceId"].encode())[:24]}'
            if (not isinstance(text, str) or not text.strip() or digest(text.encode()) != row['textSha256']
                    or row['id'] != identity or row['id'] in ids or text in texts
                    or row['sourceId'] in old_ids or text in old_texts):
                raise ValueError('Text, identity or split overlap differs.')
            ids.add(row['id'])
            texts.add(text)
            tally[row['expected']['choice']] += 1
        if dict(tally) != {label: PER_CLASS for label in SPEC['options']}:
            raise ValueError('Frozen support differs.')
    return manifest


def requests(case: dict) -> list[dict]:
    """Project only runtime input into four fixed option orders.

    The caller must authenticate the containing freeze first. No case ID,
    expected label, split or provenance is sent to a backend.
    """
    runtime = case['input']
    if set(runtime) != {'state', 'question', 'options'} or set(runtime['state']) != {'text'}:
        raise ValueError('Unexpected runtime fields.')
    if (not isinstance(runtime['state']['text'], str) or runtime['question'] != SPEC['question']
            or list(runtime['options'].items()) != list(SPEC['options'].items())):
        raise ValueError('Runtime text or task differs.')
    pairs = list(runtime['options'].items())
    return [{'state': dict(runtime['state']), 'question': runtime['question'],
             'options': dict(pairs[offset:] + pairs[:offset])} for offset in OFFSETS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify'])
    parser.add_argument('--cache', type=Path, default=Path('.local/public-data/raw'))
    parser.add_argument('--v1', type=Path, default=Path('benchmarks/public-v1'))
    parser.add_argument('--out', type=Path, default=Path('benchmarks/banking-public-v2'))
    args = parser.parse_args()
    manifest = prepare(args.cache, args.v1, args.out) if args.command == 'prepare' else verify(args.out, args.v1)
    print(json.dumps({'protocol': PROTOCOL, 'counts': manifest['counts'], 'inferencePerformed': False}, indent=2))


if __name__ == '__main__':
    main()
