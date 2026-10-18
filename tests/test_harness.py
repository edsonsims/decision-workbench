from copy import deepcopy
import json

from workbench.backends import RuleBackend, checked_decision
from workbench.harness import evaluate

CASE = {'input': {'id': 'harness-legal', 'tenantId': 'team-a',
                  'document': {'id': 'nda', 'text': 'Keep the disclosed information confidential.'}},
        'expected': {'route': 'legal', 'category': 'confidentiality'}}


class RecordingRules(RuleBackend):
    def __init__(self):
        self.states = []

    def choose(self, state, question, options):
        self.states.append(deepcopy(state))
        return super().choose(state, question, options)


def test_harness_keeps_gold_outside_runtime_and_counts_both_order_runs():
    backend = RecordingRules()
    original = deepcopy(CASE)
    summary, records = evaluate([CASE], backend)
    assert CASE == original
    assert all('expected' not in json.dumps(state) for state in backend.states)
    assert summary['primaryDecisionCalls'] == 2
    assert summary['orderProbeDecisionCalls'] == 2
    assert summary['totalDecisionCalls'] == len(backend.states) == 4
    assert summary['neuralDecisionAttempts'] == 0
    assert summary['successfulNeuralDecisions'] == 0
    assert summary['primaryStageDecisionCounts'] == {'legal_analyst': 1, 'reviewer': 1}
    assert summary['categoryStageReached'] == 1
    assert summary['categoryCorrect'] == 1
    assert records[0]['optionOrderProbe']['changedRoles'] == []


def test_wrong_raw_choice_is_scored_wrong_even_when_policy_requires_review():
    class WrongLegal(RuleBackend):
        name = 'deliberately-wrong-test-double'

        def choose(self, state, question, options):
            if state['role'] == 'legal_analyst':
                return checked_decision('payment', {key: float(key == 'payment') for key in options}, options, self.name)
            return super().choose(state, question, options)

    summary, records = evaluate([CASE], WrongLegal())
    assert summary['categoryCorrect'] == 0
    assert summary['failureCases'] == ['harness-legal']
    assert records[0]['result']['analysis']['category'] == 'payment'
    assert records[0]['result']['requiresHumanReview']
    assert summary['contractFailures'] == 0  # Guardrail success cannot inflate quality.


def test_router_abstention_is_reported_as_unreached_specialists():
    class AbstainingRouter(RuleBackend):
        def choose(self, state, question, options):
            if state['role'] == 'router':
                return checked_decision('other', {key: float(key == 'other') for key in options}, options, self.name)
            return super().choose(state, question, options)

    summary, _ = evaluate([CASE], AbstainingRouter(), routing='model')
    assert summary['categoryCases'] == 1
    assert summary['categoryStageReached'] == 0
    assert summary['categoryCorrect'] == 0
    assert summary['primaryStageDecisionCounts'] == {'router': 1}


def test_order_probe_exposes_a_position_dependent_classifier():
    class FirstOption(RuleBackend):
        name = 'position-dependent-test-double'

        def choose(self, state, question, options):
            choice = next(iter(options))
            return checked_decision(choice, {key: float(key == choice) for key in options}, options, self.name)

    summary, records = evaluate([CASE], FirstOption())
    assert summary['optionOrderCategoryChanges'] == 1
    assert 'legal_analyst' in records[0]['optionOrderProbe']['changedRoles']
