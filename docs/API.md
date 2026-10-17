# Public interfaces

The package namespace is `m1`. The public boundary is supplied features/states
and complete factor output. Research launchers, corpus loaders, historical
checkpoint envelopes and application consumers are not runtime dependencies.

## Text and feature geometry

`RawText(sentence_id, text, language)` accepts a nonempty identifier, unchanged
text and `en` or `ru`. `segment_text(text)` defines canonical character units.
Character ranges are half-open Python Unicode string offsets, not UTF-8 byte
offsets. Capacity checks explicitly say when they use UTF-8 bytes.

```python
from m1 import RawText, FeatureBatch

rows = (RawText("example-1", "Participant_A opened the container.", "en"),)
features = FeatureBatch(
    rows=rows,
    unprimed=(unprimed_units,),
    prefixes=((purchase_units, give_units, own_units, open_units,
               close_units, report_units, show_units),),
    provider_metadata={"provider_version": "your-compatible-provider"},
)
```

This is a shape/interface sketch: the tensors come from your provider, not from
the text constructor. Every tensor must be ordinary contiguous finite CPU FP32,
with shape `[len(segment_text(text)), 2048]`, detached from gradients and outside
inference-tensor mode. All raw and seven conditioned banks must have disjoint
storage. Row IDs must be unique. Even a zero-event sentence requires all seven
banks; predicted or Gold HEADs cannot select which banks are captured.

`FeatureProvider.capture(rows)` returns this complete `FeatureBatch` in the same
row order. Provider metadata is retained for inspection and is not proof of
feature origin. No external encoder, model name, network client or credentials
are chosen by the core package.

## Supplied owners

`components_from_states(states, binding_kind="full_context83")` restores the
five groups `type14`, `localizer`, `original66`, `heads`, `binding`. Use
`binding_kind="native74"` for that architecture's complete state instead.
`export_component_states(models)` creates detached copies for serialization.

The factory validates exact complete state keys, shapes, ordinary CPU FP32,
finiteness and storage separation. It derives content commitments from supplied
bytes, restores frozen modules and preserves global RNG state. Inference rejects
state mutation, execution hooks and per-instance method overrides. Supplied
geometry and hashes establish structural integrity, not a training history or
semantic competence.

`experimental_components(seed=..., binding_kind=...)` is an explicit invented
state factory for tests and interface development. **It is not a pretrained
fallback.** Real inference needs independently supplied, compatible trained
states and features.

Canonicalization and slots are separate: `canonicalizer_from_state` and
`slots_from_state` preserve their numerical owners. The default event-factor
runtime does not run application-specific consumer encoding.

The optional [example configuration](../examples/inference.example.json)
contains only supported factory options and a CPU thread setting. The core does
not read a configuration file implicitly. A caller can apply it explicitly:

```python
import json
import torch
from m1.factories import components_from_states

with open("examples/inference.example.json", encoding="utf-8") as stream:
    config = json.load(stream)
torch.set_num_threads(config["torch_num_threads"])
models = components_from_states(
    bundle.states,
    binding_kind=config["binding_kind"],
    binding_mode=config["binding_mode"],
    type_architecture=config["type_architecture"],
)
```

`bundle` is a caller-authenticated state bundle, as described below; this
configuration supplies neither trained states nor a feature provider.

## Inference

```python
from m1 import predict_features, predict_raw

result = predict_features(features, models, budget_check=check_resource_budget)
# Or capture complete features through an explicit provider first:
result = predict_raw(rows, models=models, provider=provider,
                     budget_check=check_resource_budget)
result.validate()
```

Use ordinary CPU FP32 defaults and `torch.set_num_threads(2)`. The optional
budget callback can raise to stop computation. The caller controls external
provider resources and must synchronize any mutable state before capture/save.

`compose_components(raw_rows, prefix_inputs, models, *, budget_check=None)` is
the adapter for complete unprimed rows and the native seven-candidate prefix
DTO. `predict_components(rows, *, models, provider, budget_check=None)` accepts
annotation-free `RawText` instances or dictionaries with exactly
`sentence_id`, `text`, `language`. Neither adapter loads historical model stores
or chooses an implicit provider.

`ComponentResult` retains `localization`, `factors`, all interval scores,
`binding_batches`, `saved_binding_rows`, factor weights and provenance. Calling
`validate()` checks retained inputs, owners and results for mutation.

Each factor row has a `sentence_id`, complete `events`, and directed `relations`.
An event includes its half-open character span, `TYPE`, `polarity_id` and role
mentions with original surfaces/character spans. Null roles are omitted from the
mention list. A relation names source and target event indices and `relation_id`.
Self edges and `NONE` edges are omitted; direction is preserved.

| Output | Index order |
| --- | --- |
| Types | PURCHASE, GIVE, OWN, OPEN, CLOSE, REPORT, SHOW |
| Roles | agent, source, possessor, recipient, addressee, context, owner, object |
| Polarity | NONE, AFFIRMATIVE, NEGATIVE |
| Relations | NONE, BEFORE, AFTER |

The existing polarity decision adds a uniform `0.1` AFFIRMATIVE intercept before
argmax. It is part of this implementation, not a newly calibrated probability.
Argument binding uses the native beam-8 decoder; exact optimality is claimed
only for the localization dynamic program.

`replay_saved_binding(result, reopened_rows)` checks the complete nine-array
saved banks and three factor tensors, then replays the actual decoder. Saved
score integrity is distinct from semantic correctness.

## State bundles

```python
from m1.checkpoint import save_state_bundle, load_state_bundle
from m1.factories import export_component_states

receipt = save_state_bundle(
    "new-state-directory",
    export_component_states(models),
    metadata={"binding_kind": "full_context83"},
)
# Preserve both commitments through a trusted independent channel.
restored = load_state_bundle(
    "new-state-directory",
    expected_archive_sha256=receipt["archive_sha256"],
    expected_manifest_sha256=receipt["manifest_sha256"],
)
```

The codec writes only `weights.safetensors` and `manifest.json`; it never loads
pickled/executable model objects. Both external digests authenticate complete
physical bytes before parsing. It rejects symlinked paths, unsafe state names,
missing/extra arrays, non-FP32/nonfinite values and oversized archives. Native
key order is retained; each load produces private ordinary CPU tensors.

Directories are create-only. The manifest is written last and fsynced; an
interrupted save may leave an incomplete directory which cannot load. It is not
silently resumed or overwritten. Default caps are 512 MiB for an archive and
1 MiB for its manifest. The codec does not establish compatibility with arbitrary
external checkpoints or approve a feature basis.

## Explicit compatibility domain

`legacy_domain_report(result)` reports at most two events, 96 UTF-8 bytes per
mention and 4096 UTF-8 bytes per text. `require_legacy_compatible(result)` raises
`LegacyDomainError` when those limits are exceeded; the exception retains the
complete result as `component_result`. It does not filter or clip predictions.

A successful domain check is necessary but is not proof of an external
application's dense-vector/consumer compatibility. Those formats are outside
this standalone release.

