# Feature Provider audit and local execution

Audit date: 2026-10-20. This describes the provider for **M1 Experimental 1**.
Matching feature width alone is insufficient: another encoder, tokenizer,
checkpoint revision, layer or prefix recipe changes the input basis.

## Donor identity

The donor is [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base),
revision **`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`**. The original snapshot
contains multimodal weights. M1 loads only its text model using
`Qwen3_5ForCausalLM` and the snapshot's `Qwen3_5TextConfig`, with an explicit
`model.language_model.` → `model.` key mapping. Vision and MTP parameters
are excluded. The resident text model has 1,881,825,088 unique parameters:
BF16 except the original 2,592 FP32 DeltaNet parameters. It has 24 decoder
blocks, width 2048, and vocabulary size 248320.

The model and tokenizer are frozen. No LoRA, quantization, generation, chat
template, remote model code, replacement donor or model training is used.
The provider uses generic PyTorch DeltaNet operations and SDPA full attention.

The exact upstream revision has an
[Apache-2.0 license](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/LICENSE).
Obtain the snapshot separately. It is not included in the M1 state package or
ordinary Git history. Redistributors must preserve its license and applicable
notices; the M1 MIT license does not replace upstream terms.

## Eight banks from one input

Each input is `RawText(sentence_id, text, language)` with language `en` or `ru`.
There is no Gold event span, count, role, polarity or type input.

1. Segment the unchanged original string into lexical units using
   [`canonical_units.segment_text`](../src/m1/canonical_units.py). Offsets are
   Python Unicode code points, with half-open `[start, end)` spans, not bytes or
   UTF-16 positions. Unicode tables must be version 15.0.0 (tested Python 3.12).
   Contiguous letters/marks/numbers/underscore form units. Internal apostrophes,
   curly apostrophes and hyphens join word characters. Whitespace is omitted;
   punctuation otherwise forms separate units. No Unicode normalization occurs.
2. Use the pinned fast tokenizer with `add_special_tokens=False`, without
   truncation or chat formatting. Append exactly one EOS token, ID **248044**.
   Right padding uses EOS IDs with attention mask zero. Original EOS in text
   and more than 512 valid tokens including prefix and appended EOS are rejected.
3. Encode the original text once for the unprimed bank. Encode seven separate
   strings, each `prefix[k] + original_text`, for the conditioned banks. All
   seven are computed before event localization, including event-free inputs.
4. Capture the residual output of decoder block **11, zero based**, before the
   final RMSNorm. A scoped hook exits immediately after this block: blocks 12–23
   and the language-model head do not run. The full text donor remains resident.
5. Copy captured states to CPU FP32. For each original unit, take the arithmetic
   mean of every nonempty tokenizer offset overlapping that unit. Prefix,
   appended EOS and padding states are not pooled. Every original unit,
   punctuation and suffix is retained. No representation normalization or
   learned provider projection is applied. M1's subsequent neural heads perform
   their own declared transformations.
6. Return eight separate, ordinary, detached, finite contiguous CPU FP32 tensors
   `[original_units, 2048]`. Different banks have distinct storage.

The seven prefixes use English definitions for both input languages. Their
exact form is `Candidate meaning: ` + definition + `\n\nText:\n`:

| Order | Type | Definition |
| --- | --- | --- |
| 0 | PURCHASE | Receiving ownership of an item in exchange for payment. |
| 1 | GIVE | Permanently transferring ownership of an item to another person without payment. |
| 2 | OWN | Having possession or ownership of an item as a state. |
| 3 | OPEN | Changing an object from a closed state to an open state. |
| 4 | CLOSE | Changing an object from an open state to a closed state. |
| 5 | REPORT | Communicating information or a proposition to another person. |
| 6 | SHOW | Intentionally making an item visible to another person without transferring ownership. |

Prefix alignment checks that token IDs and offsets for the original suffix
match unprefixed tokenization exactly. Prefix/source crossing tokens or changed
suffix token IDs are rejected. The provider never repairs these cases by
trimming or changing the input.

The semantic-prefix basis SHA-256 is
`e351c258c07e90a0c003d0da22337acc9d1a476ee414ec4329509c99e35df232`.
The provider runs original-text items in batches of at most eight, then
row-major seven-prefix items in batches of at most eight. Exact feature-byte
comparison requires matching this item order, batch and padding geometry;
bitwise equivalence across hardware or different batching is not promised.

## Public implementation and infrastructure extraction

The standalone implementation is
[`m1.providers.qwen35.Qwen35FeatureProvider`](../src/m1/providers/qwen35.py).
It implements the existing `FeatureProvider.capture` protocol. The donor model
is optional and is not imported or loaded when importing the ordinary M1 API.
The text CLI is [`examples/infer_text.py`](../examples/infer_text.py).

The audited research implementation already supported arbitrary annotation-free
text, but its source opener also checked historical experiment manifests and an
unrelated GTE snapshot. Those were infrastructure dependencies, not features
used by these eight banks. The public module replaces those project-root
requirements with the same exact donor-file commitments and declared numerical
recipe. It does not depend on historical corpora, private run directories,
GTE, an API server, authentication secrets or a network connection at inference.
CAN uses its separate learned byte model, not GTE embeddings.

## Install and run

Install the optional provider dependencies in addition to M1:

```bash
python -m pip install -e '.[provider]'
hf download Qwen/Qwen3.5-2B-Base \
  --revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
  --local-dir ./qwen35-2b-base
```

The provider authenticates these files before loading:

| File | SHA-256 |
| --- | --- |
| config.json | ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4 |
| merges.txt | a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d |
| model.safetensors-00001-of-00001.safetensors | 928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d |
| model.safetensors.index.json | 74d2ddfe79f10f35b27b498632f02b97b60dd9ec39b35d7c5a890c399284e319 |
| tokenizer.json | fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927 |
| tokenizer_config.json | e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de |
| vocab.json | ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003 |

After separately obtaining the experimental checkpoint and its trusted manifest
hash, run real inference:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/infer_text.py \
  --checkpoint ./m1-experimental-1 \
  --manifest-sha256 d00f28e9f8f0591a6ee465670490f45a9de9ab15194ca2d1e886fbab41b8f63c \
  --donor ./qwen35-2b-base --device cuda:0 \
  --language ru --text 'Мария закрыла зелёный шкаф после разговора.'
```

Use `--device cpu` where supported by the installed PyTorch build; CPU donor
latency has not been measured. M1 heads run on CPU with two Torch threads.
The CLI defaults to a maximum of 128 canonical units and fails on overflow;
`--max-units` explicitly changes that resource budget. The donor's 512-token
limit still applies, including each type prefix and EOS. Work for all spans
and binding candidates grows with input length, so this is not an unbounded
text service.

The optional stack used for validation is Transformers 5.15.0, tokenizers
0.22.2, huggingface-hub 1.27.0, accelerate 1.14.0 and the repository's pinned
core Torch/NumPy/safetensors versions. CUDA model execution was tested using
the CUDA 13.0 PyTorch build. Model and tokenizer loading use
`local_files_only=True` and `trust_remote_code=False`; after initial downloads
and installation the computation is local and can run offline.

## Resources and validation scope

The donor's single safetensors archive occupies **4,548,221,488 bytes**.
The text-only resident parameter payload is **3,763,655,360 bytes**; runtime
workspaces, activations and framework overhead are additional. The exact
2048-wide eight-bank FP32 payload is `65,536 × number_of_original_units` bytes,
excluding tensor/metadata overhead.

A completed historical development capture processed 368 texts (184 EN,
184 RU), 13,698 original units and 2,944 physical encodings in 368 batches:
420.979 seconds, peak process RSS 4,943,106,048 bytes, owned CUDA allocation
3,991,314,432 bytes, reserved 4,030,726,144 bytes and whole-device memory
4,859 MiB. Those figures include model loading, verification, serialization
and measurement; they are not single-request latency. The paired historical
M1/consumer evaluation used two restored owners and took 1,036.940 seconds
with peak RSS 4,354,400,256 bytes. The two phases did not retain donor VRAM
simultaneously. These measurements do not establish a production memory
minimum or throughput guarantee for the standalone CLI.

The provider fixtures exercise exact pooling, all seven views, Unicode
geometry, layer-11 early exit, ordinary tensor storage, mutation checks,
closed-provider behavior, overflow, bad suffix alignment, wrong donor bytes
and generic-kernel wrappers. They contain no pretrained assets and make no
semantic accuracy claim. [Real two-text smoke evidence](PROVIDER_SMOKE.json) records 16/16 banks exactly
equal to the audited original capture, followed by trained M1 + CAN inference
without Gold inputs. The measured standalone donor load/authentication took
6.218 seconds, capture 2.354 seconds and checkpoint load plus M1/CAN inference
1.922 seconds. Total validation wall time (including reference replay) was
10.885 seconds, peak RSS 4,838,285,312 bytes and owned CUDA allocation
3,888,609,792 bytes on an RTX 5090. These two examples establish an executable
path, not accuracy or generalization: several actual CAN IDs are wrong or
outside the finite inventory and are retained verbatim. Full M1 qualification
remains open.

A second [actual CLI check](PROVIDER_CLI_SMOKE.json) ran `examples/infer_text.py`
with the trained checkpoint and local CUDA donor on the Russian input in a
separate process. It completed in 10.175 seconds and imported no private research
modules. Both smoke checks ran offline. This confirms the public entry point;
it does not improve or certify the displayed predictions.
