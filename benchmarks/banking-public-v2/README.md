# BANKING77 card intents: development/test v2 freeze

Prepared 2026-11-01 before any model inference for this version: 160 development
queries from the official training CSV and 160 test queries from the official test
CSV, 20 per each of the same eight selected intents as public-v1.

**Attribution:** PolyAI; Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew
Henderson and Ivan Vulić (2020), “Efficient Intent Detection with Dual Sentence
Encoders.” [Official source](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b),
revision `57ec275d8078af65b7731c2a98be812d844a6d6b`, `banking_data/train.csv`
and `banking_data/test.csv`. Licensed under [CC BY 4.0](LICENSE.CC-BY-4.0.txt).

Changes: select eight intents, remove defined exact ambiguities/overlaps and
public-v1 banking IDs/text, deterministically hash-rank 20 queries per class and
split, add the unchanged task question/options plus separate original labels,
IDs, hashes and provenance. Original query text is preserved without trimming,
normalization or truncation. The full raw CSVs are not republished here.

The [protocol](../../docs/banking-public-v2.md) documents selection, observed
counts, exact-separation limits and the planned development-only abstention
method. Only data preparation, verification and runtime input projection are
implemented. There are no predictions, policy-fit results or model-quality
scores. Known DecisionTune training exposure and unknown other-model exposure
prevent a training-unseen claim. Four future rotations are not exhaustive order
invariance. Labels come from the publisher, not owner review.

```sh
python -m workbench.banking_v2 verify
```
