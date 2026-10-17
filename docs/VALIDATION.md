# Validation scope

This is implementation validation for the experimental `0.1.0a1` source
candidate. It does not qualify semantic accuracy, language coverage, trained
weights, a feature provider or a completed M1 freeze.

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
CPU execution used two threads. These are local observations; GitHub Actions is
configured but no hosted workflow result is claimed before its first run.

## Remaining evidence

All factor-quality qualifications remain open in this public release: exact
localization, conditional type identity, full-span/cross-event argument binding,
polarity on predicted spans and directed relations in the integrated pipeline.
There is no redistributable trained checkpoint, public training recipe or
independent end-to-end evaluation corpus in this candidate. Provider/model
compatibility and resource budgets require separate validation for each real
deployment. The tests above must not be presented as substitutes for that work.
