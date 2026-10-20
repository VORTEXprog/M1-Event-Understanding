# Validation scope

This document separates implementation validation, trained-state parity and
model-quality evidence. None establishes completed M1 qualification. The original
source-release checks below remain historical observations.

## Executed checks

On 2026-10-18 the candidate was built as a wheel and installed into a fresh
CPU-only environment with Python 3.12.3, PyTorch 2.13.0+cpu, NumPy 2.5.2,
safetensors 0.8.0 and pytest 9.1.1.

| Check | Observed result |
| --- | --- |
| Tests against the installed wheel | 131 passed in 69.52 seconds; no skipped or expected-failure cases |
| Isolated installed-package imports | Package and all 63 submodules imported; CUDA was not initialized |
| `feature_localization.py` against the installed wheel | Completed on invented English/Russian feature banks |
| `feature_pipeline.py` against the installed wheel | Completed through actual owners and decoders; retained nine saved binding arrays per row |
| Dependency consistency | `pip check`: no broken requirements |

The test duration is one execution observation, not a latency or throughput
benchmark. The examples' untrained predictions are interface demonstrations,
not evidence of text understanding. No GPU, network, credentials, external
encoder, trained model or non-public dataset was needed for these checks.

## What the tests establish

- Exact localization partition/MAP and gradient agreement with exhaustive
  enumeration on small invented cases.
- Full-context pooling behavior, relative-position geometry, zero-bias parity
  and uncapped multi-event localization.
- Seven-type interval scoring, feature isolation and deterministic owner
  restoration.
- Genuine native74/full-context83 composition, including empty, single-event
  and three-or-more-event outputs, complete saved arrays and decoder replay.
- Independent state storage, mutation rejection, hook/method-override rejection
  and fresh-process state restoration.
- Data-only checkpoint round trips, independent byte commitments, finite CPU
  FP32 validation, path/size restrictions and interrupted-directory rejection.
- Canonicalization/slot utilities and explicit legacy-domain errors retaining
  complete factor output.

## Reproduce locally

Install the versions documented in the README, then run:

```bash
python -m pytest
python -m compileall -q src tests examples
python -m pip check
python examples/feature_localization.py
python examples/feature_pipeline.py
```

The final wheel check deliberately ran the tests outside the checkout with
`-c /dev/null -p no:cacheprovider`, with the wheel installed and no `PYTHONPATH`.
The fresh-process test also exercises restoration in a separate interpreter.
Each example was separately run with `python -I` against the installed package.
CPU execution used two threads. These are local observations; hosted workflow
results must be checked for the particular public commit.

## Remaining evidence

All factor-quality qualifications remain open in this public release: exact
localization, conditional type identity, full-span/cross-event argument binding,
polarity on predicted spans and directed relations in the integrated pipeline.
The separately prepared [Experimental 1 checkpoint](EXPERIMENTAL_1.md) is a
pre-release candidate whose weights have not been published. It has a pinned
provider and real-feature parity evidence, but no public full training recipe or
redistributable independent end-to-end corpus. Historical RAW quality gates fail.
The implementation and parity checks must not be presented as substitutes for
semantic qualification.


## Experimental 1 preparation, 2026-10-20

A new CPU-only Python 3.12.3 environment was created and dependencies downloaded
independently. The built wheel was installed without an editable checkout.
All **140 tests passed in65.29 seconds**, with no skipped or expected-failure cases,
running outside the source tree without `PYTHONPATH`. Both original invented
examples passed under `python -I`; all 68 submodules imported without initializing
CUDA. `pip check`, compileall and local documentation-link checks passed.

The selected **trained** checkpoint was restored from that installed wheel in
another isolated process. On16 private real inputs (8 EN + 8 RU, including0–3predicted
events), all 99,288 TYPE scores,64 LOC arrays,14 HEAD96 arrays, original conditioning
outputs,144 BIND arrays, beam decisions and52 CAN outputs matched native reference
hashes. This took9.496 seconds. Neither Gold nor a donor call entered that replay.
Private corpus/features remain outside the Git and release packages.

The standalone real provider separately reproduced16/16 banks exactly for two
novel RU/EN inputs. An offline public-only CLI process then ran the trained
checkpoint successfully. Its inaccurate CAN symbols are retained in
[the actual smoke output](PROVIDER_CLI_SMOKE.json), not replaced by expected IDs.
Two earlier provider attempts failed load-time generic-kernel validation before
any donor forward; corrected validation and regression fixtures precede the
successful capture. No donor weights or numeric feature recipe changed.

[Machine-readable verification](EXPERIMENTAL_1_VERIFICATION.json),
[provider parity and resources](PROVIDER_SMOKE.json) and
[historical failing quality results](EXPERIMENTAL_1_RESULTS.json) have different
scopes. None is an independent final qualification or completed M1 freeze.
