"""Local evidence checks. None of these functions calls a decision provider."""

from __future__ import annotations

import re
from copy import deepcopy
from decimal import Decimal, localcontext
from typing import Any

_AMOUNT = re.compile(r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_CURRENCY = re.compile(r"[A-Z]{3}\Z")
_LEGAL_TERMS = {
    "confidentiality": r"confidential\w*|disclos\w*|secret\w*",
    "termination": r"terminat\w*|cancel\w*|expires?",
    "payment": r"payment\w*|invoice\w*|fees?|pay(?:ment)?",
}


def identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 128


def input_issues(task: Any) -> list[str]:
    if not isinstance(task, dict):
        return ["Input must be an object."]
    issues = []
    if set(task) - {"id", "tenantId", "document", "invoice", "ledger"}:
        issues.append(
            "Input contains unsupported fields; evaluation labels are not runtime input."
        )
    if not identifier(task.get("id")) or not identifier(task.get("tenantId")):
        issues.append(
            "Input needs nonblank id and tenantId strings of at most 128 characters."
        )
    document = task.get("document")
    if isinstance(document, dict) and isinstance(document.get("text"), str):
        try:
            if len(document["text"].encode("utf-8")) > 65536:
                issues.append("Document exceeds 65536 UTF-8 bytes.")
        except UnicodeEncodeError:
            issues.append("Document must contain valid Unicode scalar values.")
    ledger = task.get("ledger")
    if isinstance(ledger, list) and len(ledger) > 1000:
        issues.append("Ledger exceeds 1000 entries.")
    return issues


def router_input(task: dict) -> dict:
    """Keep foreign tenant records out of every provider's context."""
    result = deepcopy(task)
    if isinstance(result.get("ledger"), list):
        result["ledger"] = [
            row
            for row in result["ledger"]
            if isinstance(row, dict) and row.get("tenantId") == task.get("tenantId")
        ]
    return result


def legal_issues(document: Any) -> list[str]:
    if not isinstance(document, dict) or set(document) != {"id", "text"}:
        return ["Document must contain exactly id and text."]
    if not identifier(document["id"]) or not isinstance(document["text"], str):
        return ["Document needs a nonblank id and string text."]
    try:
        if len(document["text"].encode("utf-8")) > 65536:
            return ["Document exceeds 65536 UTF-8 bytes."]
    except UnicodeEncodeError:
        return ["Document must contain valid Unicode scalar values."]
    return []


def legal_evidence(document: dict, category: str) -> dict | None:
    """Extract a literal sentence; this proves quotation, not legal meaning."""
    text = document["text"]
    pattern = _LEGAL_TERMS.get(category)
    match = re.search(pattern, text, re.IGNORECASE) if pattern else None
    if match is None:
        return None
    start = (
        max(text.rfind("\n", 0, match.start()), text.rfind(".", 0, match.start())) + 1
    )
    while start < len(text) and text[start].isspace():
        start += 1
    ends = [
        pos + 1
        for delimiter in ("\n", ".")
        if (pos := text.find(delimiter, match.end())) >= 0
    ]
    end = min(ends) if ends else len(text)
    while end > start and text[end - 1].isspace():
        end -= 1
    return {
        "documentId": document["id"],
        "start": start,
        "end": end,
        "quote": text[start:end],
        "offsetUnit": "unicode_code_points",
    }


def verify_citation(document: dict, citation: Any) -> bool:
    if not isinstance(citation, dict):
        return False
    start, end = citation.get("start"), citation.get("end")
    text = document.get("text")
    return (
        isinstance(text, str)
        and type(start) is int
        and type(end) is int
        and 0 <= start < end <= len(text)
        and citation.get("documentId") == document.get("id")
        and citation.get("offsetUnit") == "unicode_code_points"
        and citation.get("quote") == text[start:end]
    )


def decimal_amount(value: Any) -> Decimal | None:
    # Decimal strings avoid binary float error; bounded plain notation also
    # prevents unbounded exponents from consuming memory during formatting.
    if not isinstance(value, str) or len(value) > 128 or not _AMOUNT.fullmatch(value):
        return None
    return Decimal(value)


def exact_total(values: list[Decimal]) -> Decimal:
    if not values:
        return Decimal(0)
    lowest = min(0, *(value.as_tuple().exponent for value in values))
    highest = max(0, *(value.adjusted() for value in values))
    with localcontext() as context:
        context.prec = highest - lowest + len(str(len(values))) + 3
        return sum(values, Decimal(0))


def finance_checks(task: dict) -> tuple[dict, str, dict]:
    """Return provider facts, separate policy label, and tenant-scoped evidence.

    This evaluates one invoice in one currency. It performs no FX conversion,
    monetary quantization, payment authorization or model arithmetic.
    """
    invoice, ledger = task.get("invoice"), task.get("ledger")
    issues: list[str] = []
    if not isinstance(invoice, dict) or set(invoice) != {"id", "currency", "amount"}:
        issues.append("Invoice must contain exactly id, currency and amount.")
        invoice = invoice if isinstance(invoice, dict) else {}
    if not identifier(invoice.get("id")):
        issues.append("Invoice id must be a nonblank string of at most 128 characters.")
    currency = invoice.get("currency")
    if not isinstance(currency, str) or not _CURRENCY.fullmatch(currency):
        issues.append("Invoice currency must be three uppercase letters.")
    amount = decimal_amount(invoice.get("amount"))
    if amount is None:
        issues.append(
            "Invoice amount must be a plain decimal string of at most 128 characters."
        )
    if not isinstance(ledger, list) or len(ledger) > 1000:
        issues.append("Ledger must be an array of at most 1000 entries.")
        ledger = []
    relevant = [
        row
        for row in ledger
        if isinstance(row, dict)
        and row.get("tenantId") == task["tenantId"]
        and row.get("invoiceId") == invoice.get("id")
    ]
    values: list[Decimal] = []
    for row in relevant:
        if set(row) != {"id", "tenantId", "invoiceId", "currency", "amount"} or not all(
            identifier(row.get(key)) for key in ("id", "tenantId", "invoiceId")
        ):
            issues.append("Matching ledger entry has invalid fields or identifiers.")
        if not isinstance(row.get("currency"), str) or not _CURRENCY.fullmatch(
            row["currency"]
        ):
            issues.append("Matching ledger currency must be three uppercase letters.")
        value = decimal_amount(row.get("amount"))
        if value is None:
            issues.append(
                "Matching ledger amount must be a plain decimal string of at most 128 characters."
            )
        else:
            values.append(value)
    valid = not issues
    currencies_match = all(row.get("currency") == currency for row in relevant)
    total = exact_total(values) if len(values) == len(relevant) else None
    facts = {
        "valid": valid,
        "matchingEntryCount": len(relevant),
        "currenciesMatch": currencies_match,
        "invoiceAmount": format(amount, "f") if amount is not None else None,
        "ledgerTotal": format(total, "f") if total is not None else None,
        "issues": issues,
    }
    if not valid:
        policy = "invalid"
    elif not relevant:
        policy = "missing"
    elif len(relevant) > 1:
        policy = "duplicate"
    elif not currencies_match or amount != total:
        policy = "mismatch"
    else:
        policy = "match"
    evidence = {
        "tenantId": task["tenantId"],
        "invoice": deepcopy(invoice),
        "matchingEntries": deepcopy(relevant),
        "facts": deepcopy(facts),
    }
    return facts, policy, evidence


def verify_finance_evidence(task: dict, evidence: Any, policy: str) -> bool:
    _, recomputed, expected = finance_checks(task)
    return recomputed == policy and evidence == expected
