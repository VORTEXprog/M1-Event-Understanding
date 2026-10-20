# Experimental 1: rights and distribution scope

Prepared on 2026-10-20. The project owner authorized preparation of the
experimental project weights. Uploading the weights and publishing the GitHub
pre-release still require the owner's separate confirmation.

## Project source and trained M1 weights

Project-owned source and the project-owned M1 Experimental 1 parameter tensors
are offered under the [MIT license](../LICENSE), copyright 2026 XRedMzK. Preserve
that license with a redistributed copy of the project weight package. This grant
does not cover independently obtained donor assets or third-party dependencies.

The candidate contains three data-only state bundles: 272 component tensors,
32 routed canonicalizer tensors and four optional historical slot tensors.
These are the selected learned M1 owners, not a repackaged language-model
checkpoint. Their exact identities, sizes and quality limits are recorded in
[Experimental 1](EXPERIMENTAL_1.md) and its authenticated checkpoint manifest.

The reviewed training lineage uses finite, project-authored English/Russian
templates, constructed participant/object inventories and annotated combinations.
The canonicalizer's supplemental name support includes generated nonce syllables;
its current finite inventory also includes ordinary person names and object
descriptions. Those names are task symbols, not a collected personal-records
dataset. The reviewed lineage did not identify an imported external event or
semantic-role corpus in these selected trained owners. This is a bounded
provenance finding, not a universal proof concerning every historical research
file or the rights in every possible learned output.

Training on frozen donor features does not copy the donor's parameter tensors
into these project modules. The candidate package contains only the declared M1
FP32 arrays and sanitized composition metadata. It excludes training corpora,
Gold annotations, RAW feature banks, research logs, private checkpoint envelopes,
credentials, personal filesystem paths and third-party model/tokenizer files.
Small source tests and input/output illustrations are explicitly invented.

The full historical training corpora, intermediate checkpoints and complete
training recipe are not released with Experimental 1. Consequently, restoring
and running the selected state can be reproduced; retraining its entire lineage
from scratch is not established by this release.

## Separately obtained feature donor

The compatible donor is `Qwen/Qwen3.5-2B-Base`, revision
`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`. The official repository supplies
an Apache-2.0 license at that revision. Obtain the snapshot directly from its
official source and retain its own license and applicable notices.
([Official pinned license](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/LICENSE))

The donor weights, tokenizer assets and Hugging Face library distributions are
not included in the M1 weight package or source repository. The public provider
is an M1 adapter to separately installed libraries and an authenticated local
snapshot. Details are in the [Feature Provider audit](FEATURE_PROVIDER.md) and
[dependency inventory](DEPENDENCIES.md).

M1 does not assert ownership of Qwen, relicense its assets under MIT or certify
the rights in Qwen's upstream pretraining data. Separately obtaining the official
snapshot keeps its provenance and license boundary explicit. Any later donor
redistribution would need its own inventory and preserved notices.

## Remaining publication checks

The current review identified no concrete third-party corpus or donor-parameter
file that must be removed from the selected M1 weight bundles. Before publishing,
verify the final attachment inventory against its hashes, include the project
license with the weights, and ensure the release identifies the separate donor
and dependency licenses. Final artifact and source checks must use the actual
files being uploaded; this document does not replace them.

There is no claim of production readiness, universal privacy protection or
copyright noninfringement. Model accuracy and its failed development criteria are
reported separately from licensing.
