# M1 Experimental 1

Published **Pre-release**:
[M1 Experimental 1](https://github.com/XRedMzK/M1-Event-Understanding/releases/tag/experimental-1).
This is a checkpoint of an ongoing experiment, not a new architecture release,
M1 v1.5, a completed M1 qualification, or a production-ready model. The release
targets source commit `426d13fb767fca286e960c123d84c53130e19de6`.

The archive was frozen during preparation on 2026-10-20, before publication
approval. Its enclosed documentation and `published: false` preparation metadata
record that earlier state; the published bytes and trained tensors are unchanged.
This repository's documentation records the subsequent publication.

Public inference implementation: [e18efe6](https://github.com/XRedMzK/M1-Event-Understanding/commit/e18efe6dec031aa7b5ac80f7595134c76c6f283e). This commit changes loading/provider utilities, not the trained parameter values.

## Selected trained state

The candidate uses a completed, evaluated assembly rather than the newest
unfinished experiments. Its five core groups are TYPE14, LOC13, original
conditioning66, POL/REL96 and full-context BIND83: **272 parameter tensors**.
A separate **32-tensor CAN** includes its complete frozen28 dependency and
learned4 full-context/one-blank router. The optional trained slots4 owner is
included for explicit API use; the text-to-factor CLI does not apply a two-slot
projection or discard extra events. Total: **308 tensors, 15,268,045 scalar
parameters**. All tensors are trained or retained dependencies of the evaluated
trained assembly; no randomly initialized substitute was packaged.

The three safetensors archives total **61,106,828 bytes** (58.28 MiB), excluding
JSON, documentation and the separately obtained feature donor. The public
bundle removes a redundant copy of CAN from the internal checkpoint envelope.
Native and public state digests use different key/order conventions for some
groups; physical tensor equality is checked rather than inferred from names.
The exact public commitments are in [the checksum record](EXPERIMENTAL_1_SHA256.json).

The core `components_from_states` interface still accepts exactly five groups.
`m1.experimental.load_experimental_checkpoint` loads the authenticated top-level
manifest, then restores the core, CAN32 and slots bundles separately. Old CAN96
entry points remain available; this checkpoint explicitly uses **CAN128**, not
an implicit widening or a retagged CAN96 state.

## What the numbers mean

All quality results below are **development results**, measured on the same
368 original texts (184 English, 184 Russian), containing 704 annotated events.
There are 356 distinct normalized texts and 12 consistent repeated records;
none was filtered from the metrics. Gold event counts E0/E1/E2/E3 occur in
22/106/122/118 rows. The identity vocabulary is a known finite inventory of 27
person/object IDs. These results do not establish open-domain coverage,
unseen-identity generalization, or independent final performance. The full
research corpus and private annotations are not release assets.

| Measure | All | English | Russian |
| --- | --- | --- | --- |
| LOC TP / FN / FP | 438 / 266 / 115 | 239 / 113 / 71 | 199 / 153 / 44 |
| LOC exact event-span sets | 140/368 | 87/184 | 53/184 |
| Correct event count | 182/368 | 124/184 | 58/184 |
| TYPE on identifiable predicted HEADs | 465/467 | 247/249 | 218/218 |
| Unidentifiable predicted HEADs, separate | 86 | 61 | 25 |
| Canonical BIND TP / FN / FP | 767 / 953 / 489 | 443 / 417 / 280 | 324 / 536 / 209 |
| Exact typed event and binding sets | 83/368 | 51/184 | 32/184 |
| POL, including missing/extra typed events | 422/821 | 231/425 | 191/396 |
| Exact typed event and directed REL sets | 127/368 | 78/184 | 49/184 |

LOC F1 is **0.696897**; canonical BIND F1 is **0.515457**. TYPE's conditional
fraction does not excuse missed events or the 86 unidentifiable predictions.
POL uses 704 Gold events plus 117 unmatched typed predicted events, not just
matched events. POL by native class is NONE 390/746, AFFIRMATIVE10/14 and
NEGATIVE22/61. These are the historical evaluator's denominators, unchanged.
BIND and REL exact-set denominators include every row and require the correct
typed event set. Full-span argument matching is separately 424/1720; it is not
canonical identity matching.

CAN decodes **1720/1720 Gold full argument mentions** correctly in its component
control. On actual predicted spans the whole-pipeline BIND result remains
767 true positive roles; there are 58 roles without usable canonical outputs and 12 declared
assembly-domain failures. No dictionary lookup repairs learned outputs. IDs
represent the bounded vocabulary, not general physical-instance coreference.

A frozen application-specific research consumer, excluded from the public
package, obtains **154/368** strict pair decisions, macro0.414751 and
structural0.416667. This is context for the existing full integration evidence;
it is not an executable public consumer benchmark. Missing/empty endpoints
remain incorrect under its historical strict scoring.

[Machine-readable results and source evidence hashes](EXPERIMENTAL_1_RESULTS.json)
retain all language/event-count groups and consumer denominators. No thresholds,
original results, or acceptance criteria were changed to prepare this candidate.
**The full M1 quality gates fail and M1 is not frozen.**

## Known failures

- **LOC:** missed and extra events, assertion-scope confusion, inaccurate full
  predicate boundaries and poor multi-event transfer.
- **TYPE:** 2 wrong identifiable predicted HEAD types; additional HEADs are
  unidentifiable. High conditional accuracy is not end-to-end correctness.
- **BIND:** truncated mentions, neighboring-event argument stealing, wrong roles,
  missing/extra arguments, and finite-identity limitations. Beam8 is approximate.
- **CAN:** finite 27 identity training, context/ambiguity errors, malformed or
  out-of-inventory learned IDs on bad predicted spans. Mention128/text4096 UTF-8
  byte limits remain explicit; outputs are retained and errors marked.
- **POL:** weak factual-negation and outer-scope transfer. On a separate consumed
  448 row/1344 event operator DEV, a rejected RAW-attention experiment reached
  only 673/1344 despite fitting its TRAIN. Those weights are **not** included.
- **REL:** partial temporal-pair and unrelated-third-event confusion. A rejected
  support-only replacement improved a separate504 row control but regressed
  actual368 exact sets127→113; it is **not** included.

The public API retains arbitrary predicted event counts; optional compatibility
or slot domains are not silently imposed on the component result. No calibrated
confidence, memory/dialogue application, unrestricted ontology or completed
three-origin/final qualification is supplied.

## Obtaining and running the checkpoint

Download the archive and checksum attachment from the published
[M1 Experimental 1 Pre-release](https://github.com/XRedMzK/M1-Event-Understanding/releases/tag/experimental-1):

```bash
curl --fail --location \
  https://github.com/XRedMzK/M1-Event-Understanding/releases/download/experimental-1/m1-experimental-1.tar.gz \
  --output m1-experimental-1.tar.gz
curl --fail --location \
  https://github.com/XRedMzK/M1-Event-Understanding/releases/download/experimental-1/SHA256SUMS \
  --output SHA256SUMS
sha256sum --check SHA256SUMS
printf '%s  %s\n' \
  6e72a0893767ea8206a9440d9a54697e8f051b67e8db7fe4f11022b75aea6061 \
  m1-experimental-1.tar.gz | sha256sum --check -
tar -xzf m1-experimental-1.tar.gz
```

The archive SHA-256 is
`6e72a0893767ea8206a9440d9a54697e8f051b67e8db7fe4f11022b75aea6061`.
Compare the commitment through a trusted channel independent of the downloaded
archive; matching checksums establish integrity, not model quality.

Use Python 3.12 and install the exact dependencies and feature donor described in
[Feature Provider](FEATURE_PROVIDER.md). The donor is obtained separately from
its official source, at the pinned revision. It is not bundled into M1 weights.
No substitute encoder with matching dimensions is compatible by assumption.

After extracting the checkpoint and independently verifying its top-level
manifest commitment:

```bash
python examples/infer_text.py \
  --checkpoint ./m1-experimental-1 \
  --manifest-sha256 d00f28e9f8f0591a6ee465670490f45a9de9ab15194ca2d1e886fbab41b8f63c \
  --donor ./qwen35-2b-base \
  --device cuda:0 --language en --text 'Cedric opened the trunk.'
```

Use `--language ru` for Russian. CPU donor execution is an explicit alternative;
CUDA is needed for the measured GPU path, not the core factor owners. The CLI
accepts raw text only, captures all eight real feature banks, runs trained core
owners, then canonicalizes predicted role spans. It does not accept Gold spans,
counts, types or identities. Input-budget violations fail explicitly, with no
cropping or manufactured response. Output JSON contains actual predictions;
quality is bounded by the failures above.

## Reproducibility and rights

[Feature Provider audit](FEATURE_PROVIDER.md) specifies model/revision, file
commitments, tokenization, canonical units, overlap alignment, exact prefixes,
layer and dtype policy. [Rights and package boundary](EXPERIMENTAL_1_RIGHTS.md)
describes project source/weights and separately obtained dependencies. No
closed corpus, personal paths, credentials, private research report or donor
weight file belongs in a release asset.

[Validation](VALIDATION.md) separates invented implementation tests from trained
state/feature parity and historical model-quality evidence. Full independent
benchmark replication is limited by the undistributed research corpora;
restore/parity and fresh-text smoke results must not be called qualification.
The final release notes and local preparation report record completed checks.


## Completed preparation checks

The clean installed wheel passed140 tests in65.29 seconds and all 68 submodule
imports without CUDA initialization. A fresh isolated process restored all308
trained tensors and matched native reference scores/decisions on16 real fixed
inputs, including52 CANoutputs (9.496 seconds). The standalone provider matched
16/16 real banks on two new texts; the separate offline CLI passed10.175 seconds.
Measured combined donor/M1 smoke peakRSS 4.84 GB and owned CUDA 3.89 GB apply only to
those short inputs on the tested RTX 5090, not a minimum hardware specification.
See [verification](EXPERIMENTAL_1_VERIFICATION.json),
[provider measurement](PROVIDER_SMOKE.json) and
[unaltered actual CLI predictions](PROVIDER_CLI_SMOKE.json).
