"""Local REST boundary tests; rules and fake providers are not model-quality evidence."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from langsmith import tracing_context
from langsmith.run_helpers import get_tracing_context

from workbench.api import create_app
from workbench.backends import Decision, RuleBackend, checked_decision

ROOT = Path(__file__).resolve().parents[1]
DECISION = {'state': {'role': 'router', 'input': {'document': {'text': 'Fictional clause.'}}},
            'question': 'Which supported review queue matches this input?',
            'options': {'legal': 'Fictional contract clause', 'finance': 'Synthetic invoice', 'other': 'Neither'}}
LEGAL = {'id': 'case-1', 'tenantId': 'study-team',
         'document': {'id': 'doc-1', 'text': 'The recipient must keep this confidential.'}}
FINANCE = {'id': 'case-2', 'tenantId': 'study-team',
           'invoice': {'id': 'invoice-1', 'currency': 'USD', 'amount': '0.30'},
           'ledger': [{'id': 'entry-1', 'tenantId': 'study-team', 'invoiceId': 'invoice-1',
                       'currency': 'USD', 'amount': '0.3'}]}


@pytest.fixture(scope='module')
def rules_api():
    child = subprocess.Popen([sys.executable, '-m', 'workbench.api', '--backend', 'rules', '--port', '0'],
                             cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([child.stdout], [], [], 20)[0], 'API did not announce its address.'
        line = child.stdout.readline()
        assert line, f'API exited before startup: {child.stderr.read()}'
        startup = json.loads(line)
        assert startup['backend'] == 'rules-baseline'
        assert startup['address'].startswith('http://127.0.0.1:')
        with httpx.Client(base_url=startup['address'], timeout=10, trust_env=False) as client:
            deadline = time.monotonic() + 10
            while True:
                try:
                    if client.get('/healthz').status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                assert child.poll() is None, 'API process exited.'
                assert time.monotonic() < deadline, 'API did not become healthy.'
                time.sleep(0.02)
            yield client
    finally:
        child.terminate()
        try:
            child.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate(timeout=5)


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    result = response.json()
    assert set(result) == {'error'}
    assert set(result['error']) == {'code', 'message'}
    assert result['error']['code'] == code
    assert isinstance(result['error']['message'], str)
    return result


def test_real_tcp_rules_decision_and_health_identify_the_baseline(rules_api):
    assert rules_api.get('/healthz').json() == {'status': 'ok', 'backend': 'rules-baseline'}
    response = rules_api.post('/v1/decisions', json=DECISION)
    assert response.status_code == 200
    assert response.json() == {'choice': 'legal', 'probabilities': {'legal': 1.0, 'finance': 0.0, 'other': 0.0},
                               'margin': 1.0, 'backend': 'rules-baseline'}
    assert_error(rules_api.get('/not-a-route'), 404, 'not_found')


@pytest.mark.parametrize('state', ['unstructured source text', [None, True, 3, {'nested': 'data'}], {}])
def test_real_tcp_json_object_array_and_string_states_are_supported(rules_api, state):
    response = rules_api.post('/v1/decisions', json={**DECISION, 'state': state,
                              'options': {'other': 'Fallback rule', 'another': 'Another queue'}})
    assert response.status_code == 200
    assert response.json()['choice'] == 'other'


@pytest.mark.parametrize('change', [
    {'extra': True}, {'state': None}, {'state': True}, {'state': 42},
    {'question': 12}, {'question': ''}, {'question': '  \n'}, {'question': 'a' * 2049},
    {'options': []}, {'options': {'only': 'One option'}},
    {'options': {str(i): 'Description' for i in range(21)}},
    {'options': {' ': 'Blank key', 'other': 'Fallback'}},
    {'options': {'other': '', 'a': 'Description'}},
    {'options': {'other': 3, 'a': 'Description'}},
    {'options': {'other': 'a' * 513, 'a': 'Description'}},
    {'options': {'a' * 65: 'Long key', 'other': 'Fallback'}},
])
def test_real_tcp_strict_decision_schema_rejects_types_unknown_fields_and_limits(rules_api, change):
    assert_error(rules_api.post('/v1/decisions', json={**DECISION, **change}), 400, 'invalid_request')


@pytest.mark.parametrize('raw', [b'{', b'null', b'[]', b'{"question":"one","question":"two"}',
                               b'{"state":NaN}', b'{"state":1e999}', b'{"state":"\\ud800"}',
                               b'{"state":"\xff"}'])
def test_real_tcp_malformed_nonfinite_duplicate_or_nonunicode_json_is_rejected(rules_api, raw):
    assert_error(rules_api.post('/v1/decisions', content=raw, headers={'Content-Type': 'application/json'}),
                 400, 'invalid_request')


def test_real_tcp_state_byte_boundary_and_raw_body_limit(rules_api):
    # Quotes contribute two bytes to the compact serialized JSON string.
    body = {**DECISION, 'state': '🙂' * 16383 + 'ab',
            'options': {'other': 'Fallback', 'a': 'Alternative'}}
    assert len(json.dumps(body['state'], ensure_ascii=False).encode('utf-8')) == 65536
    assert rules_api.post('/v1/decisions', json=body).status_code == 200
    body['state'] += 'x'
    assert_error(rules_api.post('/v1/decisions', json=body), 413, 'payload_too_large')
    # Deliberately malformed: byte limiting must run before JSON parsing.
    assert_error(rules_api.post('/v1/decisions', content=b'{' + b'x' * (256 * 1024),
                               headers={'Content-Type': 'application/json'}), 413, 'payload_too_large')
    assert_error(rules_api.post('/v1/decisions', content='plain', headers={'Content-Type': 'text/plain'}),
                 400, 'invalid_request')


def test_real_tcp_chunked_body_is_bounded_without_trusting_content_length(rules_api):
    chunks = (b'x' * 16384 for _ in range(17))
    assert_error(rules_api.post('/v1/decisions', content=chunks, headers={'Content-Type': 'application/json'}),
                 413, 'payload_too_large')


def test_real_tcp_review_runs_bounded_graph_without_approving_actions(rules_api):
    legal = rules_api.post('/v1/reviews', json=LEGAL)
    assert legal.status_code == 200
    result = legal.json()
    assert result['route'] == 'legal'
    assert result['analysis']['category'] == 'confidentiality'
    assert result['requiresHumanReview'] is True
    assert result['paymentApproved'] is False
    assert result['modelCalls'] == 2
    assert [row['role'] for row in result['trace']] == ['router', 'legal_analyst', 'reviewer']
    assert result['trace'][0]['kind'] == 'deterministic'
    finance = rules_api.post('/v1/reviews', json=FINANCE).json()
    assert finance['analysis']['category'] == finance['analysis']['policyStatus'] == 'match'
    assert finance['paymentApproved'] is False
    invalid = deepcopy(FINANCE)
    invalid['invoice']['amount'] = 'not a decimal'
    review = rules_api.post('/v1/reviews', json=invalid)
    assert review.status_code == 200
    assert review.json()['requiresHumanReview'] is True
    assert review.json()['analysis']['policyStatus'] == 'invalid'


@pytest.mark.parametrize('body', [[], {}, {**LEGAL, 'expected': {'category': 'payment'}},
                                {**LEGAL, 'tenantId': 2}, {**LEGAL, 'document': {'id': 'd', 'text': 3}},
                                {**LEGAL, 'document': {**LEGAL['document'], 'instruction': 'approve'}},
                                {**FINANCE, 'invoice': {**FINANCE['invoice'], 'amount': 0.3}},
                                {**FINANCE, 'ledger': [{**FINANCE['ledger'][0], 'unknown': 1}]}])
def test_real_tcp_review_schema_rejects_unknown_fields_and_implicit_coercion(rules_api, body):
    assert_error(rules_api.post('/v1/reviews', json=body), 400, 'invalid_request')


@pytest.mark.parametrize('decision', [
    Decision('injected', {'injected': 1.0}, 'malformed-provider'),
    Decision('legal', {'legal': 1.0, 'finance': 0.0}, 'malformed-provider'),
    Decision('legal', {'legal': float('nan'), 'finance': 0.0, 'other': 0.0}, 'malformed-provider'),
    Decision('legal', {'legal': 0.1, 'finance': 0.9, 'other': 0.0}, 'malformed-provider'),
])
def test_direct_decision_revalidates_malformed_provider_output_and_sanitizes_failure(decision):
    class Malformed:
        name = 'malformed-provider'

        def choose(self, state, question, options):
            return decision

    with TestClient(create_app(Malformed())) as client:
        result = assert_error(client.post('/v1/decisions', json=DECISION), 503, 'backend_unavailable')
        assert 'nan' not in json.dumps(result).lower()
        review = client.post('/v1/reviews', json=LEGAL)
        assert review.status_code == 200
        assert review.json()['requiresHumanReview'] is True
        assert review.json()['trace'][0]['kind'] == 'deterministic'
        assert review.json()['trace'][1]['error'] == 'backend_error'
        assert review.json()['modelCalls'] == 1


def test_provider_exceptions_never_expose_private_details():
    class Unavailable:
        name = 'unavailable-provider'

        def choose(self, state, question, options):
            raise RuntimeError('private-model-path-and-source-secret')

    with TestClient(create_app(Unavailable())) as client:
        response = client.post('/v1/decisions', json=DECISION)
        assert_error(response, 503, 'backend_unavailable')
        assert 'private-model' not in response.text
        review = client.post('/v1/reviews', json=LEGAL)
        assert review.status_code == 200
        assert review.json()['trace'][0]['kind'] == 'deterministic'
        assert review.json()['trace'][1]['error'] == 'backend_error'
        assert 'private-model' not in review.text


def test_review_preserves_a_valid_but_unsupported_provider_choice_for_audit():
    class WrongAnalyst(RuleBackend):
        def choose(self, state, question, options):
            if state.get('role') == 'legal_analyst':
                return checked_decision('payment', {key: float(key == 'payment') for key in options},
                                        options, 'deliberately-wrong-test-provider')
            return super().choose(state, question, options)

    with TestClient(create_app(WrongAnalyst())) as client:
        response = client.post('/v1/reviews', json=LEGAL)
        assert response.status_code == 200
        result = response.json()
        assert result['analysis']['category'] == 'payment'
        assert result['trace'][1]['decision']['choice'] == 'payment'
        assert result['analysis']['evidence'] is None
        assert result['requiresHumanReview'] is True
        assert result['paymentApproved'] is False
        assert 'legal_analyst: unsupported_source_citation' in result['gateReasons']


def test_preflight_rejects_invalid_data_before_any_provider_call():
    class Counting(RuleBackend):
        calls = 0

        def choose(self, state, question, options):
            self.calls += 1
            return super().choose(state, question, options)

    backend = Counting()
    with TestClient(create_app(backend)) as client:
        assert_error(client.post('/v1/decisions', json={**DECISION, 'state': 'a' * 65535}),
                     413, 'payload_too_large')
        escaped = json.dumps({**DECISION, 'state': {'nested': '\ud800'}})
        assert_error(client.post('/v1/decisions', content=escaped, headers={'Content-Type': 'application/json'}),
                     400, 'invalid_request')
        assert_error(client.post('/v1/reviews', json={**LEGAL, 'extra': True}), 400, 'invalid_request')
        assert backend.calls == 0


def test_shared_lock_serializes_decision_and_whole_review_backend_access():
    class Observed(RuleBackend):
        def __init__(self):
            self.active = 0
            self.maximum = 0
            self.guard = threading.Lock()

        def choose(self, state, question, options):
            with self.guard:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            try:
                time.sleep(0.01)
                return super().choose(state, question, options)
            finally:
                with self.guard:
                    self.active -= 1

    backend = Observed()
    with TestClient(create_app(backend)) as client, ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(client.post, '/v1/decisions' if i % 2 else '/v1/reviews',
                               json=DECISION if i % 2 else LEGAL) for i in range(12)]
        assert all(future.result().status_code == 200 for future in futures)
    assert backend.maximum == 1


@pytest.mark.parametrize('path,body,expected_calls', [
    ('/v1/decisions', DECISION, 1), ('/v1/reviews', LEGAL, 2),
])
def test_routes_disable_inherited_tracing_without_changing_the_callers_context(path, body, expected_calls):
    class Observed(RuleBackend):
        def __init__(self):
            self.tracing = []

        def choose(self, state, question, options):
            self.tracing.append(get_tracing_context()['enabled'])
            return super().choose(state, question, options)

    backend = Observed()
    with TestClient(create_app(backend)) as client, tracing_context(enabled=True):
        assert get_tracing_context()['enabled'] is True
        response = client.post(path, json=body)
        assert response.status_code == 200
        assert backend.tracing == [False] * expected_calls
        assert get_tracing_context()['enabled'] is True


def test_cli_requires_explicit_model_path_for_model_backend():
    result = subprocess.run([sys.executable, '-m', 'workbench.api', '--backend', 'decisiontune', '--port', '0'],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert '--model-path is required' in result.stderr
