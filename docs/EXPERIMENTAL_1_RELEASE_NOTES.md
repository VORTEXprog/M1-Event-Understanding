# M1 Experimental 1 — candidate release notes

**Pre-release. Unpublished; owner approval required before upload.**
Proposed tag: `experimental-1`. This is a public experimental trained checkpoint,
not M1 v1.5, a stable release or completed M1 qualification.

## Included

Completed compatible TYPE14/LOC13/conditioning66/POL-REL96/BIND83 states, a
separate trained CAN32 input128 router and its frozen28 dependency, and optional
slots4. Three data-only safetensors archives contain308 tensors,
15,268,045 parameters and61,106,828 weight bytes. No random substitute, donor
weights, dataset, research report or application consumer is included.

The source adds an authenticated public checkpoint loader, explicit CAN128/32
support, a pinned Qwen3.5-2B-Base feature provider and a raw-text JSON CLI. Existing
five-group and CAN96 APIs remain available. All model responses use actual
trained states and real donor features.

## Results and failures

The full adaptive-development integration retains368 texts/704 Gold events
(184 texts/language; E0/E1/E2/E3=22/106/122/118).

- LOC:438 TP,266 FN,115 FP; F1.696897; exact140/368; correct count182/368.
- TYPE:465/467 identifiable predicted HEADs;86 unidentifiable spans separate.
- BIND:767 TP,953 FN,489 FP canonicalroles; F1.515457; exact83/368.
- POL:422/821 Gold plus extra typed events; NONE 390/746,AFF 10/14,NEG 22/61.
- REL:127/368 exact typed-event-and-directed-edge sets.
- CAN:1720/1720 on Gold full mentions only; this is not end-to-end accuracy.
- External frozen research consumer (not shipped):154/368 strict pairs.

Full M1 quality gates **fail**. Known errors include assertion scope, missing and
extra events, truncated and neighboring-event arguments, finite identity
coverage, factual negation and partial temporal order. No threshold or result
was changed for release. [Model card](EXPERIMENTAL_1.md) and
[machine-readable results](EXPERIMENTAL_1_RESULTS.json) define exact denominators,
language groups and evidence commitments.

## Requirements and execution

Python 3.12; pinned PyTorch 2.13,NumPy 2.5.2,safetensors 0.8; provider extras and exact
separately downloaded Qwen revision. Core M1 owners run CPU FP32; the measured
provider path uses CUDA and mixed BF16/FP32. CPU donor execution is available but
has no measured latency guarantee. See the full
[provider/resource audit](FEATURE_PROVIDER.md) before selecting hardware.

Use `python examples/infer_text.py --help`, then the complete pinned-checkpoint
command in [the model card](EXPERIMENTAL_1.md). No Gold input is accepted.
[Independent checksums](EXPERIMENTAL_1_SHA256.json) authenticate the manifest and
three archives. Donor, source and dependency licensing are described in
[the rights audit](EXPERIMENTAL_1_RIGHTS.md).

## Limits of reproducibility

A fresh local installation can restore trained states and run new text after
obtaining the pinned files. Historical full DEV metrics are reported, with
source-evidence digests, but the undistributed research corpus prevents an
independent rerun of that entire evaluation from release assets alone. Fixed
real-feature and native/public parity checks are implementation evidence, not
an independent benchmark or final qualification. Final preparation/test results
will accompany the candidate before the owner decides whether to publish.


## Completed candidate checks

Clean installed-wheel tests:140 PASS / 0 FAIL / 0 SKIP in65.29 seconds; both invented
examples and all 68 submodule imports PASS, no CUDA initialization on import.
Independent fresh-process trained-state replay matches all native references
on16 real RU/EN inputs and52 CANoutputs. Real provider16/16 bank parity and separate
offline public-only CLI PASS; see [verification](EXPERIMENTAL_1_VERIFICATION.json).
These checks establish executable restoration and parity, not semantic quality.
