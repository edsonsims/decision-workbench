"""Decision providers. A deterministic baseline is never labelled model inference."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

MODEL_ID = 'decision-tune/decisiontune-1.0'
MODEL_REVISION = '27056cb08bcf2b3ba1fd9bccb782b4dfd0498e1f'
MANIFEST_SHA256 = '2ec93545ef3fcc2135e8b5c14e4b4690ff99535e9bc8ed5b85c98d071564a715'


@dataclass(frozen=True)
class Decision:
    choice: str
    probabilities: Mapping[str, float]
    backend: str

    @property
    def margin(self) -> float:
        ordered = sorted(self.probabilities.values(), reverse=True)
        return ordered[0] - ordered[1]

    def as_dict(self) -> dict:
        return {'choice': self.choice, 'probabilities': dict(self.probabilities),
                'margin': self.margin, 'backend': self.backend}


def checked_decision(choice: str, probabilities: dict, options: dict[str, str], backend: str) -> Decision:
    if not isinstance(backend, str) or not backend.strip() or len(backend) > 128:
        raise ValueError('A backend name must be a nonblank string of at most 128 characters.')
    if not 2 <= len(options) <= 20 or not all(isinstance(k, str) and k for k in options):
        raise ValueError('A decision needs 2 to 20 named options.')
    if set(probabilities) != set(options) or choice not in options:
        raise ValueError('The decision must use exactly the supplied options.')
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
           for p in probabilities.values()):
        raise ValueError('Decision probabilities must be finite numbers in [0,1].')
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-5):
        raise ValueError('Decision probabilities must sum to one.')
    if probabilities[choice] + 1e-6 < max(probabilities.values()):
        raise ValueError('The selected option must have a maximal probability.')
    return Decision(choice, MappingProxyType(dict(probabilities)), backend)


class DecisionBackend(Protocol):
    name: str

    def choose(self, state: Any, question: str, options: dict[str, str]) -> Decision: ...


class RuleBackend:
    """A deliberately simple transparent baseline, not an AI model."""
    name = 'rules-baseline'

    def choose(self, state: Any, question: str, options: dict[str, str]) -> Decision:
        role = state.get('role') if isinstance(state, dict) else None
        if role == 'router':
            task = state.get('input', {})
            choice = ('legal' if 'document' in task else
                      'finance' if 'invoice' in task else 'other')
        elif role == 'legal_analyst':
            text = state.get('document', {}).get('text', '').lower()
            choice = ('confidentiality' if any(word in text for word in ('confidential', 'disclos', 'secret'))
                      else 'termination' if any(word in text for word in ('terminat', 'cancel', 'expires'))
                      else 'payment' if any(word in text for word in ('payment', 'invoice', 'fees', 'pay '))
                      else 'other')
        elif role == 'finance_analyst':
            checks = state.get('checks', {})
            if not checks.get('valid'):
                choice = 'invalid'
            elif checks.get('matchingEntryCount') == 0:
                choice = 'missing'
            elif checks.get('matchingEntryCount', 0) > 1:
                choice = 'duplicate'
            else:
                try:
                    equal = Decimal(checks['invoiceAmount']) == Decimal(checks['ledgerTotal'])
                except (KeyError, InvalidOperation, TypeError):
                    equal = False
                choice = 'match' if checks.get('currenciesMatch') and equal else 'mismatch'
        elif role == 'reviewer':
            choice = 'evidence_supported' if state.get('sourceVerified') else 'needs_review'
        else:
            choice = 'other'
        if choice not in options:
            raise ValueError(f'Baseline has no rule for this decision role: {role}')
        # These one-hot values encode a rule's result. They are not confidence
        # estimates and are reported separately from neural model scores.
        return checked_decision(choice, {key: float(key == choice) for key in options}, options, self.name)


class DecisionTuneBackend:
    """Load verified local weights only; downloading is a separate explicit step."""
    name = 'decisiontune-1.0'

    def __init__(self, model_path: str | Path, backend: str = 'mlx'):
        path = Path(model_path)
        if not path.is_dir():
            raise ValueError('Provide a downloaded local DecisionTune model directory.')
        manifest = path / 'manifest.json'
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != MANIFEST_SHA256:
            raise ValueError('The model manifest differs from the pinned DecisionTune 1.0 manifest.')
        if version('decision-tune') != '1.0.0':
            raise ValueError('This adapter is tested with decision-tune==1.0.0.')
        from decision_tune import DecisionModel
        self.model = DecisionModel.from_pretrained(str(path), backend=backend)
        self.runtime_backend = backend

    def choose(self, state: Any, question: str, options: dict[str, str]) -> Decision:
        result = self.model.choose(state, question, options)
        return checked_decision(result['choice'], result['probabilities'], options, self.name)


class ReorderedBackend:
    """Probe option-order sensitivity without changing labels or descriptions."""
    def __init__(self, wrapped: DecisionBackend):
        self.wrapped = wrapped
        self.name = wrapped.name

    def choose(self, state: Any, question: str, options: dict[str, str]) -> Decision:
        return self.wrapped.choose(state, question, dict(reversed(list(options.items()))))
