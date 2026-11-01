# BANKING77 v2: freeze first, measure later

This incremental study freezes **160 development queries and 160 test queries**
for the same eight card-support intents used in public-v1. It implements an
offline selector, a pinned integrity verifier and four expected-free runtime
input projections. **No model has been evaluated on this version.** Threshold
fitting, prediction collection, abstention metrics and model comparisons remain
planned. The original public-v1 data, reports and weak adapter results are intact.

## Question and sequence

The next question is whether a development-selected score-margin gate, combined
with agreement across fixed option orders, reduces answered errors by sending
more cases to review. That is a hypothesis, not an observed result. Scores are
uncalibrated and the proposed empirical development constraint will not guarantee
a test error rate. Review means a proposed queue, not actual human review.

The sequence is deliberately staged:

1. Freeze source revisions, split membership, prompts, options, exclusions and
   the future decision rule. This data-preparation milestone is runnable now.
2. Pin actual model assets, adapter code and execution protocol before inference.
   Record all four calls per row, including errors; never pass gold or IDs to a
   backend. This execution machinery and its results remain planned.
3. Fit the fixed threshold grid using development predictions only. Require four
   valid agreeing choices, permit at most one error per ten accepted development
   rows, maximize accepted count and break ties with the highest threshold.
   Abstain-all is a valid outcome. Bind the selected policy to the exact
   development-prediction, model, adapter and protocol hashes.
4. Apply that frozen policy once to the test predictions. Keep all planned rows,
   failures and all order probes in their denominators; report answered correct,
   answered wrong, reviewed, failed and disagreeing counts. Answered error is
   undefined when nothing is answered. Any method revision needs a new version.

Steps 2–4 are protocol commitments for later work, not implemented features or
measurements in this release. There is no calibrated confidence, promised
coverage, error improvement, provider support or model-superiority claim.

## Sources and exact separation

Both CSVs come from the [official PolyAI dataset repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b)
at revision `57ec275d8078af65b7731c2a98be812d844a6d6b`.
The source [README](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/README.md#license)
applies its [CC BY 4.0 license](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE)
to the datasets. Attribution and a full license copy accompany the
[derived freeze](../benchmarks/banking-public-v2/).

Development uses the official `train.csv`; test uses the official `test.csv`.
First inspect identical text across **all 77 labels and both complete sources**,
not just the selected classes. Remove blank text, cross-label exact ambiguity,
same-label duplicate text within each source (keep the earliest CSV row), every
development text appearing anywhere in the official test source, all 80 prior
public-v1 banking source IDs, and exact text from either public-v1 task.
Preserve Unicode, whitespace, original row IDs and text hashes. There is no
length filter, truncation, semantic deduplication or model-output selection.

Within each split and class, take the 20 smallest hashes of canonical
`[protocol, split, sourceId, textSha256]`, with source ID as tie-breaker.
Insufficient support aborts preparation; it never reduces the requested count.

| Observation | Development | Test |
| --- | ---: | ---: |
| Full original source rows |10,003|3,080|
| Eligible selected-label rows after exclusions |1,079|240|
| Prior-v1 source IDs excluded |0|80|
| Frozen queries |160|160|
| Queries per selected class |20|20|

In these pinned sources, the selected labels required no additional exact
ambiguity, duplicate or train/test-text exclusion. The implementation still
enforces those rules and tests counterexamples. This absence of **exact** overlap
does not establish semantic independence, customer/conversation independence or
absence from model training. BANKING77 is explicitly a declared DecisionTune 1.0
training source, including the development and test source family here. Exposure
for other models is unknown. Public test text was available to this project's
data-preparation code; “untouched test” means no local prediction-driven tuning,
not a secret or externally administered test set.

This is balanced eight-intent supplied-query classification. It is not the full
77-intent benchmark, out-of-scope detection, live banking or payment execution.
The publisher supplied the labels; no owner annotation or professional review is
claimed. CUAD is outside this incremental v2 scope.

## Offline commands and integrity

The selector has no download path. Supply the two raw CSVs from the pinned
official URLs as `banking77-train.csv` and `banking77-test.csv` in a chosen cache.
Their SHA-256 values are checked **before parsing**:

| File | SHA-256 |
| --- | --- |
| train.csv | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| test.csv | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |

```sh
# Rebuild from explicitly supplied verified raw data; no network or inference.
python -m workbench.banking_v2 prepare --cache .local/public-data/raw

# Authenticate committed data without raw CSVs, network or model dependencies.
python -m workbench.banking_v2 verify

python -m pytest tests/test_banking_v2.py
```

The manifest binds both case files, source and prior-v1 digests, complete
selection definition, prompts, ordered options and observed class counts. Its
hash is pinned in code, so editing data and updating its manifest cannot reseal
this version. Preparation refuses different existing output files. These checks
authenticate bytes against this repository version; they are not an independent
attestation. Verification without raw CSVs cannot independently reconstruct
publisher annotations; offline preparation from both pinned CSVs does that.

`requests(case)` returns only `state`, `question` and `options`. The four option
orders rotate by 0, 1, 3 and 5, preserving label/description pairs and insertion
order. These are four rotations out of many possible permutations; they do not
establish universal order invariance. Authenticate the containing freeze before
using the projection. IDs, expected labels, split and source metadata stay evaluator-side.
There is no inference command in this module.

## Recorded preparation verification

On 2026-11-01, Python 3.13.1 on macOS arm64 passed the full 304-test suite,
including the 28 v2 tests; these are overlapping counts, not 332 distinct tests.
The original 130-case freeze and all three existing prediction reports also
verified and recomputed offline. This was verification of prior results, not new
inference. An independent implementation reconstructed both v2 case files from
the pinned CSVs and matched their bytes.

A locally built 0.2.0 wheel installed with the unchanged test lock reproduced
preparation, verification and exact replay outside the checkout using isolated
Python mode under macOS process network denial. A loopback connection probe
raised `PermissionError`; the installed verifier module matched the source hash.
The wheel is a local build, not a registry release. The case files are supplied
alongside it using explicit `--out` and `--v1` paths; they are not wheel package
data. No model weights, neural calls, quality measurements or human review were
part of these checks. CI authenticates recorded data and runs contract tests; it
does not evaluate a model.
