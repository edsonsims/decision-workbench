from copy import deepcopy
import json
import math

import pytest

from workbench.benchmark_metrics import classification_metrics, paired_option_order_metrics


LABELS = ['a', 'b', 'c']


def row(case_id, expected, prediction=None, **extra):
    return {'id': case_id, 'expected': expected, 'prediction': prediction, **extra}


def metrics(rows, **kwargs):
    return classification_metrics(rows, LABELS, bootstrap_samples=100, **kwargs)


def test_hand_computed_macro_f1_failures_and_timing_are_not_dropped():
    rows = [row('1', 'a', 'a', elapsedSeconds=1), row('2', 'a', 'b', elapsedSeconds=3),
            row('3', 'b', 'b'), row('4', 'c', 'c', error='timeout', elapsedSeconds=5),
            row('5', 'c'), row('6', 'b', 'outside')]
    original = deepcopy(rows)
    result = metrics(rows)
    assert rows == original
    assert result['cases'] == 6 and result['completed'] == 3 and result['failed'] == 3
    assert result['correct'] == 2 and result['accuracy'] == pytest.approx(1 / 3)
    assert result['completionCoverage'] == .5
    assert result['conditionalAccuracy'] == pytest.approx(2 / 3)
    assert result['perClass']['a']['f1'] == pytest.approx(2 / 3)
    assert result['perClass']['b']['f1'] == .5
    assert result['perClass']['c']['f1'] == 0
    assert result['macroF1'] == pytest.approx(7 / 18)
    assert result['perClass']['b']['falseNegative'] == 1
    assert result['perClass']['c']['failedPredictions'] == 2
    assert result['failureReasons'] == {'exception': 1, 'invalid_prediction': 1, 'missing_prediction': 1}
    assert sum(sum(x.values()) for x in result['confusion'].values()) == 3
    assert result['timing']['samples'] == 3
    assert result['timing']['meanSeconds'] == 3
    assert result['timing']['medianSeconds'] == 3
    assert result['timing']['p95Seconds'] == pytest.approx(4.8)
    json.dumps(result, allow_nan=False)


def test_absent_fixed_label_remains_in_macro_denominator():
    result = metrics([row('1', 'a', 'a')])
    assert result['accuracy'] == 1 and result['macroF1'] == pytest.approx(1 / 3)
    assert result['perClass']['b']['support'] == 0


@pytest.mark.parametrize('prediction', [None, 'outside', [], {'choice': 'a'}, True, 1])
def test_invalid_and_missing_predictions_are_failed_cases(prediction):
    result = metrics([row('1', 'a', prediction)])
    assert result['accuracy'] == result['macroF1'] == result['completionCoverage'] == 0
    assert result['conditionalAccuracy'] is None
    assert result['perClass']['a']['falseNegative'] == 1


def test_bootstrap_is_seeded_order_invariant_and_warns_on_degeneracy():
    rows = [row(str(i), 'a', 'a' if i % 3 else 'b') for i in range(20)]
    first = metrics(rows, seed=123)
    assert first == metrics(list(reversed(rows)), seed=123)
    assert first['bootstrap']['seed'] == 123
    for key in ('accuracy', 'macroF1', 'completionCoverage'):
        interval = first['bootstrap']['intervals'][key]
        assert 0 <= interval['low'] <= interval['high'] <= 1
    assert not first['bootstrap']['intervals']['accuracy']['degenerate']
    assert first['bootstrap']['intervals']['completionCoverage']['degenerate']
    assert 'misleading' in first['bootstrap']['caveat']


def test_cluster_bootstrap_keeps_related_cases_together():
    rows = [row('a1', 'a', 'a', group='document-a'), row('a2', 'a', 'a', group='document-a'),
            row('b1', 'b', 'a', group='document-b'), row('b2', 'b', 'a', group='document-b')]
    clustered = metrics(rows, cluster_key='group')
    assert clustered['bootstrap']['units'] == 2
    assert clustered['bootstrap']['resamplingUnit'] == 'group'
    assert clustered['bootstrap']['intervals']['accuracy']['low'] == 0
    assert clustered['bootstrap']['intervals']['accuracy']['high'] == 1
    assert metrics(rows)['bootstrap']['units'] == 4


def test_paired_changes_align_ids_and_report_completion_denominators():
    primary = [row('1', 'a', 'a'), row('2', 'b', 'a'), row('3', 'c'), row('4', 'a', 'a')]
    changed = [row('4', 'a', error='timeout'), row('3', 'c', 'c'), row('2', 'b', 'b'), row('1', 'a', 'b')]
    result = paired_option_order_metrics(primary, changed, LABELS, bootstrap_samples=100, seed=7)
    assert result['pairs'] == 4 and result['bothCompleted'] == 2
    assert result['bothCompletedCoverage'] == .5
    assert result['choiceChanges'] == 2 and result['conditionalChoiceChangeRate'] == 1
    assert result['completionChanges'] == 2
    assert result['correctToIncorrect'] == result['incorrectToCorrect'] == 2
    assert result['accuracyDelta'] == 0
    assert result == paired_option_order_metrics(list(reversed(primary)), list(reversed(changed)),
                                                  LABELS, bootstrap_samples=100, seed=7)


def test_both_failed_pairs_do_not_claim_order_stability():
    rows = [row('1', 'a', error='timeout')]
    result = paired_option_order_metrics(rows, rows, LABELS, bootstrap_samples=0)
    assert result['bothCompleted'] == 0 and result['conditionalChoiceChangeRate'] is None
    assert result['choiceChanges'] == 0
    assert result['accuracyDelta'] == 0
    assert result['bootstrap']['method'] == 'disabled'


def test_paired_bootstrap_uses_same_cases_in_both_runs():
    rows = [row(str(i), LABELS[i % 3], LABELS[(i + i % 2) % 3]) for i in range(12)]
    result = paired_option_order_metrics(rows, rows, LABELS, bootstrap_samples=100)
    for metric in ('accuracyDelta', 'macroF1Delta', 'completionCoverageDelta'):
        assert result[metric] == 0
        assert result['bootstrap']['intervals'][metric] == {'low': 0, 'high': 0, 'degenerate': True}


@pytest.mark.parametrize('rows', [[], [None], [row('1', 'a'), row('1', 'b')],
                                    [row('', 'a')], [row('1', 'unknown')],
                                    [row('1', 'a', error='')], [row('1', 'a', error=True)],
                                    [row('1', 'a', extra='untracked')]])
def test_invalid_rows_rejected(rows):
    with pytest.raises(ValueError):
        metrics(rows)


@pytest.mark.parametrize('elapsed', [-1, math.nan, math.inf, True, '1', 10 ** 1000])
def test_invalid_timings_rejected(elapsed):
    with pytest.raises(ValueError):
        metrics([row('1', 'a', 'a', elapsedSeconds=elapsed)])


def test_finite_individual_timings_cannot_overflow_report():
    with pytest.raises(ValueError, match='sum'):
        metrics([row('1', 'a', 'a', elapsedSeconds=1e308), row('2', 'a', 'a', elapsedSeconds=1e308)])


@pytest.mark.parametrize('labels', [[], ['a', 'a'], [''], [True], 'abc', {1, 2}])
def test_invalid_fixed_labels_rejected(labels):
    with pytest.raises(ValueError):
        classification_metrics([row('1', 'a', 'a')], labels)


@pytest.mark.parametrize('options', [{'bootstrap_samples': True}, {'bootstrap_samples': -1},
                                     {'bootstrap_samples': 100001}, {'seed': True},
                                     {'confidence': math.nan}, {'confidence': 0},
                                     {'confidence': 1}, {'cluster_key': 'arbitrary'},
                                     {'cluster_key': 'group'}])
def test_invalid_bootstrap_configuration_rejected(options):
    with pytest.raises(ValueError):
        classification_metrics([row('1', 'a', 'a')], LABELS, **options)


@pytest.mark.parametrize('other', [[row('2', 'a')], [row('1', 'b')],
                                     [row('1', 'a', group='other')]])
def test_paired_ids_labels_and_groups_must_match(other):
    with pytest.raises(ValueError):
        paired_option_order_metrics([row('1', 'a')], other, LABELS)
