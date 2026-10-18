import math

import pytest
from hypothesis import given, strategies as st

from workbench.backends import RuleBackend, checked_decision

OPTIONS = {'a': 'First option', 'b': 'Second option'}


@given(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
def test_valid_probabilities_preserve_mass_and_immutable_snapshot(probability):
    values = {'a': probability, 'b': 1 - probability}
    winner = max(values, key=values.get)
    decision = checked_decision(winner, values, OPTIONS, 'property-test')
    values['a'] = -1
    assert math.isclose(sum(decision.probabilities.values()), 1)
    assert 0 <= decision.margin <= 1
    with pytest.raises(TypeError):
        decision.probabilities['a'] = 0


@pytest.mark.parametrize('values,choice', [
    ({'a': float('nan'), 'b': 0.5}, 'a'),
    ({'a': float('inf'), 'b': 0}, 'a'),
    ({'a': True, 'b': 0}, 'a'),
    ({'a': -0.1, 'b': 1.1}, 'b'),
    ({'a': 0.4, 'b': 0.4}, 'a'),
    ({'a': 0.1, 'b': 0.9}, 'a'),
    ({'a': 1}, 'a'),
    ({'a': 0.5, 'b': 0.5, 'injected': 0}, 'a'),
    ({'a': 0.5, 'b': 0.5}, 'injected'),
])
def test_invalid_model_outputs_are_rejected_before_policy_decisions(values, choice):
    with pytest.raises(ValueError):
        checked_decision(choice, values, OPTIONS, 'invalid-provider')


@pytest.mark.parametrize('backend', [None, '', '   ', object(), 'x' * 129])
def test_provider_metadata_must_remain_json_serializable(backend):
    with pytest.raises(ValueError):
        checked_decision('a', {'a': 1, 'b': 0}, OPTIONS, backend)


def test_finance_baseline_reads_tool_facts_not_a_supplied_answer():
    options = {key: key for key in ('match', 'mismatch', 'missing', 'duplicate', 'invalid')}
    decision = RuleBackend().choose({'role': 'finance_analyst', 'checks': {
        'valid': True, 'matchingEntryCount': 1, 'currenciesMatch': True,
        'invoiceAmount': '0.30', 'ledgerTotal': '0.31', 'status': 'match',
    }}, 'Choose a review queue.', options)
    assert decision.choice == 'mismatch'
    assert decision.backend == 'rules-baseline'


def test_baseline_is_separate_from_ai_and_option_order_does_not_change_its_rule():
    backend = RuleBackend()
    options = {'legal': 'Contract document', 'finance': 'Invoice', 'other': 'Insufficient evidence'}
    state = {'role': 'router', 'input': {'document': {'text': 'An agreement.'}}}
    assert backend.choose(state, 'Route this task.', options).choice == 'legal'
    assert backend.choose(state, 'Route this task.', dict(reversed(list(options.items())))).choice == 'legal'
