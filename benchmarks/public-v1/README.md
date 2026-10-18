# public-v1: attributed public classification slices

This freeze contains **130 labeled examples**: 50 exact CUAD contract excerpts across five categories and 80 BANKING77 queries across eight intents. Each category has ten examples, selected deterministically from official test sources before inference. The [dataset guide](../../docs/datasets.md) describes the adaptation, exclusions, exposure caveats, and reproduction commands.

## Data license and attribution

The selected source texts and labels are distributed under **[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)**. A [full license copy](LICENSE.CC-BY-4.0.txt) accompanies this slice. The repository's software license does not replace this data license. No endorsement by the original authors is implied.

**CUAD:** The Atticus Project; Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*, NeurIPS Datasets and Benchmarks, 2021. [Paper](https://arxiv.org/abs/2103.06268), [publisher and CC BY 4.0 statement](https://www.atticusprojectai.org/cuad/), [pinned source](https://github.com/The-Atticus-Project/cuad/tree/67faa0e6023b04fcaae6cc09497ab00e5d63a2a2).

**BANKING77:** PolyAI; Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić. *Efficient Intent Detection with Dual Sentence Encoders*, NLP for Conversational AI, ACL, 2020. [Paper](https://arxiv.org/abs/2003.04807), [pinned source](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b), [source license](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE).

**Changes in this study:** select five CUAD and eight banking labels; exclude identical-text ambiguity and same-label duplicates; choose ten examples per label by fixed hash rank; convert CUAD's supplied answer spans into a clause-topic classification task; add authored questions, option descriptions, source metadata, and checksums. Source text is preserved exactly, without truncation. This is not the full CUAD extraction or 77-intent BANKING77 benchmark.

## Files and integrity

- `cases.jsonl`: source text in `input.state.text`; fixed questions/options; original labels only in evaluator-side `expected` and source metadata.
- `manifest.json`: source revisions, licenses, selection protocol, counts, groups, exact case IDs, and cases checksum.
- `LICENSE.CC-BY-4.0.txt`: license text copied from the pinned PolyAI source, applicable to the selected CC BY 4.0 data.

Run `python -m workbench.datasets verify` from the repository root. Verification requires neither raw archives nor network access. The manifest identity is pinned in the preparation module. CUAD's 50 spans come from **34 contracts**; preserve `groupId` when estimating uncertainty.

BANKING77 is an explicitly disclosed DecisionTune 1.0 training source. CUAD exposure is not proven absent. These slices do not establish training-unseen generalization, legal correctness, account safety, or commercial performance. See the [exposure discussion](../../docs/datasets.md#training-exposure).
