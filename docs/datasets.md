# Public datasets and the public-v1 freeze

This study adds two real public labeled tasks alongside the original authored workflow fixtures. It evaluates **classification of supplied text**, using the same `state`, `question`, and `options` backend interface. It does not send these examples through the fictional-clause/reconciliation graph or treat them as legal advice, account access, or payment decisions.

## Sources and licenses

| Dataset | Official source and pinned revision | License evidence | Adaptation |
| --- | --- | --- | --- |
| CUAD | [The Atticus Project repository](https://github.com/The-Atticus-Project/cuad/tree/67faa0e6023b04fcaae6cc09497ab00e5d63a2a2), `data.zip`, revision `67faa0e6023b04fcaae6cc09497ab00e5d63a2a2` | The [publisher's CUAD page](https://www.atticusprojectai.org/cuad/) explicitly identifies CC BY 4.0 | Five-category classification of exact annotated answer spans from `test.json` |
| BANKING77 | [PolyAI repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b), `banking_data/test.csv`, revision `57ec275d8078af65b7731c2a98be812d844a6d6b` | The pinned [README](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/README.md#license) applies its [CC BY 4.0 license](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE) to its datasets | Eight-category card/support intent classification from the official test split |

Attribution, full license text, and the exact derived-data changes accompany the committed [public-v1 slice](../benchmarks/public-v1/README.md). Raw archives stay under ignored `.local/public-data/raw/`; preparation verifies their SHA-256 values before parsing. No private contracts, account records, or employer data are used.

CUAD's original task is answer-span extraction across 41 categories. Here, an annotated span is already supplied and only five categories are offered: anti-assignment, audit rights, cap on liability, non-compete, and termination for convenience. There are no absent-clause or arbitrary-text negatives. Results therefore concern this selected-label classification adaptation, not contract discovery, legal correctness, or the full CUAD benchmark.

BANKING77 has 77 intents. This slice offers eight related card/payment intents, including delivery, general/contactless card failures, declined/pending payments, unexpected fees, and unrecognized payments. Results do not measure full 77-intent classification, out-of-scope detection, or an operational banking service.

## Training exposure

The [pinned DecisionTune 1.0 model card](https://huggingface.co/decision-tune/decisiontune-1.0/blob/27056cb08bcf2b3ba1fd9bccb782b4dfd0498e1f/README.md) explicitly lists **BANKING77** in fine-tuning. Its official test split is useful for this reproducible comparison, but it is not evidence of performance on an unseen dataset family. The same disclosure applies even when no local training occurs.

CUAD is not named in that declared fine-tuning list. This is not proof that its contracts or similar text were absent from base-model pretraining, benchmark development, or other exposure. Other comparison models need their own exposure assessment; lack of a disclosed overlap is not a clean-room claim.

## Selection protocol

The protocol was fixed before running inference on this slice. The committed manifest records its definition and exact case identities. Preparation uses no model output.

1. Read only the official test sources. For each CUAD annotation, verify that its text exactly equals the source context at `answer_start`; preserve its text and original document identifier.
2. Consider identical text across **all** source categories, including categories outside the selected options. Exclude selected examples whose identical text has multiple source labels. Remove blank text and repeated same-label text; keep the lexicographically first source ID when identical duplicates exist.
3. Within each selected class, rank candidates by SHA-256 of the canonical JSON array `[protocolId, datasetId, sourceId, textSha256]`. Break a hash tie by source ID and take the first ten.
4. Keep all chosen text intact. There is **no character/token length filter and no truncation**. If a model cannot accept a chosen case, retain that case in the denominator and record its failure.

Selection changes prevalence: every selected class has ten examples. This is a small balanced study, not a population estimate. New prompts, options, or case selections require a new protocol/version and fresh results rather than rewriting public-v1 after observing errors.

| Slice | Source candidates | Outside selected labels | Ambiguous exact-text annotations removed | Eligible selected examples | Frozen examples | Groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUAD | 2,643 annotated spans from the test source | 2,183 | 45 | 415 | 50 | 34 contracts |
| BANKING77 | 3,080 test queries | 2,760 | 0 | 320 | 80 | 80 query IDs |

No blank or same-label duplicate text was removed from the selected classes in these pinned sources. CUAD eligible counts are 124 anti-assignment, 102 audit rights, 104 cap on liability, 46 non-compete, and 39 termination for convenience. Each selected BANKING77 intent has 40 eligible queries. The longest frozen CUAD excerpt is 911 characters/UTF-8 bytes; the longest frozen banking query is 241. These observed lengths are not selection gates.

CUAD excerpts from the same contract are not independent observations. Use `groupId` for document-clustered uncertainty estimates, and report that the 50 excerpts represent only 34 contracts. BANKING77 query IDs are record groups, not proven customer identities or independent conversation clusters.

## Reproduce and verify

From the repository root:

```sh
# Downloads only the two pinned sources if they are missing from the cache.
python -m workbench.datasets prepare

# Rebuild from verified local raw files, with no network.
python -m workbench.datasets prepare --offline

# Verify committed texts, prompts, options, labels, IDs and manifest without raw data.
python -m workbench.datasets verify
```

Existing prepared files are only accepted if their bytes already match; preparation refuses to overwrite a different file. A changed raw cache is rejected rather than silently replaced. Default paths are `.local/public-data/raw/` and `benchmarks/public-v1/`. `--cache` and `--out` can select other paths. `--per-class` produces an explicit custom slice, but `verify` authenticates only the published ten-per-class public-v1 freeze.

The manifest binds the cases file and includes source revisions/checksums, questions, options, selection rules, counts, source IDs, document groups, and labels. Its own SHA-256 is pinned in `workbench/datasets.py`, so changing cases and editing their manifest hash together does not reseal this version. The freeze records:

- Cases SHA-256: `07de590102887155a3228d4662899403d393e7e2b65eb060c098bcee21eba552`
- Manifest SHA-256: `9e98775052d6ddf7e9a723dc07b71e3b7f6ba62b92062cf53306f96f02458cf2`

This detects changes relative to the published repository version. It is not a cryptographic signature or independent dataset attestation.

## Module boundary

```python
from pathlib import Path
from workbench.datasets import load_public_cases, verify_prepared

root = Path("benchmarks/public-v1")
manifest = verify_prepared(root)  # offline; authenticate the complete freeze first
cases = load_public_cases(root / "cases.jsonl")
for case in cases:
    runtime = case["input"]
    # The backend receives only runtime; expected remains evaluator-side.
    # decision = backend.choose(**runtime)
    expected_choice = case["expected"]["choice"]
```

Each case includes `id`, `task`, `dataset`, `split`, `sourceId`, `sourceSha256`, `groupId`, `input`, `expected`, and `metadata`. Only `input` enters the backend. It contains `state={"text": ...}`, a fixed question, and the fixed task options. `expected.choice` comes from the original CUAD annotation category or banking CSV label; metadata retains that source label, text checksum, document ID, and CUAD offset separately.

`load_public_cases` validates schema-sensitive runtime fields, source-label mapping, and text checksums. Call `verify_prepared` before evaluating the published slice to also authenticate exact rows and their order. These functions never load weights or execute inference.

The [offline tests](../tests/test_datasets.py) exercise Unicode offsets, ambiguity exclusion, stable stratification, no truncation, corrupt/downloaded cache handling, reproducible preparation, label leaks, and attempts to alter both data and manifest.
