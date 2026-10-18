"""Fixed-label metrics; failures stay in the denominator. No score calibration.

Rows must cover every planned case, including exceptions/timeouts. This module
cannot detect cases a caller omitted. Bootstrap intervals describe empirical
resampling uncertainty, not contamination, annotation or population validity.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
import random
from statistics import fmean


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _labels(labels):
    if (not isinstance(labels, Sequence) or isinstance(labels, (str, bytes))
            or not labels or any(not isinstance(x, str) or not x.strip() for x in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError('labels must be a nonempty sequence of distinct nonblank strings.')
    return tuple(labels)


def _rows(rows, labels, cluster_key):
    if cluster_key not in (None, 'group'):
        raise ValueError("cluster_key must be None or 'group'.")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError('rows must be a nonempty sequence covering all planned cases.')
    parsed, ids = [], set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) - {
                'id', 'expected', 'prediction', 'error', 'elapsedSeconds', 'group'}:
            raise ValueError('Each row must be a mapping with only documented fields.')
        case_id, expected = row.get('id'), row.get('expected')
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError('Case ids must be nonblank, unique strings.')
        if not isinstance(expected, str) or expected not in labels:
            raise ValueError('Each expected label must belong to the fixed labels.')
        error = row.get('error')
        if error is not None and (not isinstance(error, str) or not error.strip()):
            raise ValueError('error must be a nonblank string or None.')
        prediction = row.get('prediction')
        failure = ('exception' if error is not None else
                   'missing_prediction' if prediction is None else
                   'invalid_prediction' if not isinstance(prediction, str) or prediction not in labels else None)
        elapsed = row.get('elapsedSeconds')
        if elapsed is not None and (not _finite(elapsed) or elapsed < 0):
            raise ValueError('elapsedSeconds must be finite and nonnegative, or None.')
        group = row.get('group')
        if (group is not None or cluster_key) and (not isinstance(group, str) or not group.strip()):
            raise ValueError('Clustered rows need a nonblank group string.')
        parsed.append({'id': case_id, 'expected': expected,
                       'prediction': None if failure else prediction,
                       'failure': failure, 'elapsed': elapsed, 'group': group})
        ids.add(case_id)
    return sorted(parsed, key=lambda row: row['id'])


def _scores(rows, labels):
    confusion = {label: {choice: 0 for choice in labels} for label in labels}
    failures = Counter()
    support = Counter(row['expected'] for row in rows)
    for row in rows:
        if row['failure']:
            failures[row['expected']] += 1
        else:
            confusion[row['expected']][row['prediction']] += 1
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = support[label] - tp  # Includes errors and absent/invalid predictions.
        per_class[label] = {
            'support': support[label], 'truePositive': tp, 'falsePositive': fp,
            'falseNegative': fn, 'failedPredictions': failures[label],
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / support[label] if support[label] else 0.0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    correct = sum(row['prediction'] == row['expected'] for row in rows)
    completed = sum(row['failure'] is None for row in rows)
    return {'correct': correct, 'completed': completed,
            'accuracy': correct / len(rows), 'completionCoverage': completed / len(rows),
            'macroF1': fmean(value['f1'] for value in per_class.values()),
            'perClass': per_class, 'confusion': confusion}


def _quantile(values, probability):
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[math.ceil(index)] * fraction


def _bootstrap(rows, statistic, *, bootstrap_samples, seed, confidence, cluster_key):
    if type(bootstrap_samples) is not int or not 0 <= bootstrap_samples <= 100000:
        raise ValueError('bootstrap_samples must be an integer from 0 to 100000.')
    if type(seed) is not int:
        raise ValueError('seed must be an integer.')
    if not _finite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must be finite and strictly between zero and one.')
    units = {}
    for row in rows:
        key = row[cluster_key] if cluster_key else row['id']
        units.setdefault(key, []).append(row)
    population = [units[key] for key in sorted(units)]
    rng, distributions = random.Random(seed), {}
    for _ in range(bootstrap_samples):
        sample = [row for _ in population for row in rng.choice(population)]
        for name, value in statistic(sample).items():
            distributions.setdefault(name, []).append(value)
    tail = (1 - confidence) / 2
    return {
        'method': 'percentile' if bootstrap_samples else 'disabled',
        'resamples': bootstrap_samples, 'seed': seed, 'confidence': confidence,
        'resamplingUnit': 'group' if cluster_key else 'case', 'units': len(population),
        'intervals': {name: {'low': _quantile(values, tail),
                             'high': _quantile(values, 1 - tail),
                             'degenerate': min(values) == max(values)}
                      for name, values in distributions.items()},
        'caveat': ('Percentile intervals assume representative independent resampling units. '
                   'Small or convenience samples, few clusters and degenerate intervals can be misleading. '
                   'They do not establish population coverage, label quality or absence of contamination.'),
    }


def classification_metrics(rows, labels, *, bootstrap_samples=2000, seed=42,
                           confidence=0.95, cluster_key=None):
    """Summarize all planned rows against fixed labels, with zero-division F1=0.

    A row is {id, expected, prediction?, error?, elapsedSeconds?, group?}.
    Any error overrides a supplied prediction. Missing/unknown predictions are
    failures, not a new label. Confusion contains completed outputs only; failed
    predictions are separately counted and contribute false negatives. All fixed
    labels (including absent ones) contribute to macro F1. Optional group-level
    bootstrap resamples whole groups and preserves the case-weighted estimand.
    """
    labels = _labels(labels)
    rows = _rows(rows, labels, cluster_key)
    scores = _scores(rows, labels)
    elapsed = [row['elapsed'] for row in rows if row['elapsed'] is not None]
    if not _finite(sum(elapsed)):
        raise ValueError('The sum of supplied timings must also be finite.')
    result = {'schemaVersion': 1, 'labels': list(labels), 'cases': len(rows), **scores,
              'failed': len(rows) - scores['completed'],
              'failureReasons': dict(sorted(Counter(row['failure'] for row in rows if row['failure']).items())),
              'conditionalAccuracy': scores['correct'] / scores['completed'] if scores['completed'] else None,
              'timing': {'samples': len(elapsed), 'totalSeconds': sum(elapsed),
                         'meanSeconds': fmean(elapsed) if elapsed else None,
                         'medianSeconds': _quantile(elapsed, .5) if elapsed else None,
                         'p95Seconds': _quantile(elapsed, .95) if elapsed else None,
                         'scope': 'Only supplied timings; includes completed and failed attempts.'},
              'definitions': {'accuracy': 'correct / all planned cases',
                              'macroF1': 'unweighted mean of fixed-label F1; zero divisions are 0',
                              'completionCoverage': 'valid in-label prediction without error / all cases',
                              'conditionalAccuracy': 'correct / completed; never substitute for end-to-end accuracy'}}

    def statistic(sample):
        values = _scores(sample, labels)
        return {key: values[key] for key in ('accuracy', 'macroF1', 'completionCoverage')}

    result['bootstrap'] = _bootstrap(rows, statistic, bootstrap_samples=bootstrap_samples,
                                     seed=seed, confidence=confidence, cluster_key=cluster_key)
    return result


def paired_option_order_metrics(primary_rows, perturbed_rows, labels, *, bootstrap_samples=2000,
                                seed=42, confidence=0.95, cluster_key=None):
    """Compare identical cases after option reordering; retain failed pairs.

    Pair by id, never list position. Both-completed changes are conditional on
    completion and have an explicit denominator. Completion transitions and
    paired accuracy differences include every planned pair. The same indices
    (or clusters) are resampled for both runs.
    """
    labels = _labels(labels)
    primary = _rows(primary_rows, labels, cluster_key)
    perturbed = _rows(perturbed_rows, labels, cluster_key)
    if [row['id'] for row in primary] != [row['id'] for row in perturbed]:
        raise ValueError('Paired runs must contain exactly the same case ids.')
    for left, right in zip(primary, perturbed):
        if left['expected'] != right['expected'] or left['group'] != right['group']:
            raise ValueError('Paired expected labels and groups must agree.')
    pairs = [{'id': a['id'], 'group': a['group'], 'primary': a, 'perturbed': b}
             for a, b in zip(primary, perturbed)]

    def counts(sample):
        both = [p for p in sample if not p['primary']['failure'] and not p['perturbed']['failure']]
        changed = sum(p['primary']['prediction'] != p['perturbed']['prediction'] for p in both)
        def a_correct(pair):
            return pair['primary']['prediction'] == pair['primary']['expected']

        def b_correct(pair):
            return pair['perturbed']['prediction'] == pair['perturbed']['expected']

        first = _scores([p['primary'] for p in sample], labels)
        second = _scores([p['perturbed'] for p in sample], labels)
        return {'pairs': len(sample), 'bothCompleted': len(both), 'choiceChanges': changed,
                'bothCompletedCoverage': len(both) / len(sample),
                'conditionalChoiceChangeRate': changed / len(both) if both else None,
                'completionChanges': sum(bool(p['primary']['failure']) != bool(p['perturbed']['failure']) for p in sample),
                'correctToIncorrect': sum(a_correct(p) and not b_correct(p) for p in sample),
                'incorrectToCorrect': sum(not a_correct(p) and b_correct(p) for p in sample),
                'accuracyDelta': second['accuracy'] - first['accuracy'],
                'macroF1Delta': second['macroF1'] - first['macroF1'],
                'completionCoverageDelta': second['completionCoverage'] - first['completionCoverage']}

    def statistic(sample):
        values = counts(sample)
        return {key: values[key] for key in ('accuracyDelta', 'macroF1Delta',
                                            'completionCoverageDelta', 'bothCompletedCoverage')}

    return {'schemaVersion': 1, **counts(pairs), 'deltaDirection': 'perturbed minus primary',
            'bootstrap': _bootstrap(pairs, statistic, bootstrap_samples=bootstrap_samples,
                                    seed=seed, confidence=confidence, cluster_key=cluster_key)}
