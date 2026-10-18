# Research challenges

M1 is an experimental, MIT-licensed Python toolkit for extracting structured event information from current-turn text. Its numerical components cover event localization (LOC), event type (TYPE), argument-role binding (BIND), polarity (POL), and directed event relations (REL). We welcome **independent technical criticism**, small reproducible counterexamples, and carefully designed research proposals.

**Important release boundary:** this is a **source-only research preview**, not a pretrained language-understanding model. The repository does **not** include trained checkpoints, an operational text feature provider, a public training recipe, or an independent semantic-quality evaluation set. Its examples use invented features and untrained states. Passing numerical/implementation tests does **not** demonstrate that M1 understands text. See [validation scope](VALIDATION.md) and [public interfaces](API.md).

## Where independent review would help

### 1. LOC — localization across unseen constructions ([Issue #1](https://github.com/XRedMzK/M1/issues/1))

**Problem:** identifying the correct event interval on previously unseen linguistic constructions, especially when candidate intervals compete with the span-level NULL alternative or with overlapping event intervals.

**What exists:** the implementation scores every canonical half-open interval across seven event types, combines conditional type evidence with an event-versus-NULL score, and uses **exact dynamic programming** to choose a non-overlapping event set. Exact search is not proof that its input scores generalize.

**Useful investigations:**

- Inspect original-character-unit geometry and span alignment; provide a minimal counterexample if you find a defect.
- For a fixed score bank, compare the best legal event set **including** a specified reference event with the best set **excluding** it. A positive local gain is not necessarily a positive global margin.
- Propose a rights-cleared RU/EN evaluation with construction-family holdouts, separately reporting familiar and unfamiliar formulations.

**Code:** [canonical unit segmentation](../src/m1/canonical_units.py), [span/NULL scoring](../src/m1/span_context_assertion_null_v1.py), [relative-context scoring](../src/m1/span_relative_context_assertion_null_v1.py), [event-set loss and exact decoding](../src/m1/typed_event_set_localization_loss_v1.py).

### 2. BIND — argument-role scoring and transfer ([Issue #2](https://github.com/XRedMzK/M1/issues/2))

**Problem:** assigning the right argument spans to the right event and role, including multiword mentions, multiple events, and legitimately shared participants.

**What exists:** BIND enumerates **all structurally valid original-unit start/end pairs**; it scores them using event-conditioned representations, role interaction, and explicit NULL choices. It then uses an **approximate width-8 beam decoder** for eight roles. Thus, "not in the final beam" does *not* mean "never represented as an initial span candidate."

**Useful investigations:**

- Analyze the scoring objective before designing a new proposer or reranker. Where available, compare the reference path's score with the selected path's score under the **same conditional scoring law**.
- Build invented-score counterexamples tracing at which role/beam step a path is lost; distinguish *search error* from *scoring/model error*.
- Examine how role conditioning handles neighboring events and legitimate shared arguments. Do not impose a global one-argument/one-event constraint.

Some historical internal experiments reported scoring/transfer failures and no gain from a wider beam, but their trained weights and private datasets are **not included** here; those results cannot currently be reproduced from this repository alone.

**Code:** [complete span enumeration and scorer](../src/m1/joint_argument_span_score_v1.py), [joint role potentials and beam-8 decoder](../src/m1/structured_joint_role_argument_v1.py), [event-conditioned binding](../src/m1/event_conditioned_pair_owner_v1.py), [full-context binding](../src/m1/full_context_parallel_binding_owner_v1.py).

### 3. E3 — three-or-more-event composition ([Issue #3](https://github.com/XRedMzK/M1/issues/3))

**Problem:** keeping event identities and argument ownership distinct as text contains three or more interacting events. A participant may correctly fill roles in more than one event, and long spans must remain intact.

**Evidence boundary:** an internal three-event stress test reported **0/64 whole-example exact matches** for its evaluated candidates. That is a result for **one internal test**, **not** a public, independently reproducible benchmark or a general statement about M1's accuracy. Whole-example exact match becomes harder as the number of required decisions increases; evaluate per-event and per-role metrics alongside it.

**Useful investigations:**

- Review the difference between independent per-event processing and actual cross-event information exchange.
- Design tests for shared arguments, same-type competing events, and long mentions, while keeping the annotation policy explicit.
- Propose controlled ablations and separate E1/E2/E3+ scores, avoiding automatic negative labels for arguments of neighboring events.

**Code:** [cross-event role context](../src/m1/cross_event_role_context_v1.py), [full-context binding](../src/m1/full_context_parallel_binding_owner_v1.py), [structured role scoring](../src/m1/structured_joint_role_argument_v1.py), [component pipeline](../src/m1/runtime.py).

## How to contribute

You do **not** need access to private training artifacts to make a useful contribution. Good starting points include:

1. **Review an existing equation, algorithm, or interface** and point to the exact source line with a reasoned counterexample.
2. **Write a small regression test** using invented text, synthetic FP32 tensors, or explicitly redistributable fixtures. State the expected invariant.
3. **Propose an experiment** with a falsifiable hypothesis, baseline/control, fixed data and model budgets, evaluation metrics, and estimated compute cost.
4. **Help design a reproducible public evaluation** using data that can lawfully be redistributed and independently held-out construction groups.

For numerical reproduction, follow the [installation and testing steps in the README](../README.md) and [validation document](VALIDATION.md). Tests exercise source-code behavior, not semantic accuracy. Before changing an architecture or public interface, discuss the change in the matching issue. Include results **and negative findings** when available.

### Research and security boundaries

- Do not submit private prompts, credentials, personally identifying information, restricted corpora, checkpoint files, or generated research artifacts.
- Do not optimize against hidden qualification material or change acceptance thresholds to manufacture a pass.
- Clearly distinguish measured results from predictions or hypotheses; report sample sizes and uncertainty on small slices.
- Respect the rights of upstream code, models and datasets. MIT on M1-owned code does not relicense external assets.
- See [contribution guidelines](../CONTRIBUTING.md), [third-party notices](../THIRD_PARTY_NOTICES.md), and [security policy](../SECURITY.md).

**The goal is to make M1 easier to critique and improve, not to claim a solved understanding model before the evidence exists.**
