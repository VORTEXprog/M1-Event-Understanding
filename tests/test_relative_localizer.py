"""Real all-span localizer arithmetic on invented, nonproduction FP32 states."""
import math

import pytest
import torch

from m1 import span_relative_context_assertion_null_v1 as head


def invented_state():
    # A private deterministic stream initializes a structural fixture only.
    generator = torch.Generator(device="cpu").manual_seed(3161)
    state = {
        name: (torch.randn(shape, generator=generator) * .015).contiguous()
        for name, shape in head.STATE_SHAPES.items()
    }
    state[head.BIAS_KEY].zero_()
    return state


def inputs():
    raw = torch.sin(torch.arange(7 * 4 * 2048, dtype=torch.float32) * .007)
    evidence = torch.cos(torch.arange(10 * 7, dtype=torch.float32) * .09)
    return raw.reshape(7, 4, 2048).contiguous(), evidence.reshape(10, 7).contiguous()


def test_zero_relative_bias_preserves_all_native_attention_scores_and_decisions():
    state = invented_state()
    native = {name: state[name].clone() for name in head.baseline.STATE_KEYS}
    before = torch.get_rng_state().clone()
    old = head.baseline.SpanConditionedContextAssertionNullHead(native)
    new = head.RelativeSpanContextAssertionNullHead(state)
    raw, evidence = inputs()
    left, right = old(raw, evidence), new(raw, evidence)
    for name in ("type_logp", "event_logits", "span_null_logits", "pair_indices"):
        assert torch.equal(getattr(left, name).view(torch.uint8), getattr(right, name).view(torch.uint8))
    for gold in ((), ((0, 1, 0),), ((0, 1, 0), (2, 4, 2))):
        assert torch.equal(head.negative_log_likelihood(left, gold), head.negative_log_likelihood(right, gold))
    assert head.predict(left) == head.predict(right)
    assert torch.equal(before, torch.get_rng_state())


def test_all_31_signed_relative_buckets_match_scalar_rule_and_translate():
    n = 130
    pairs = head.exact.complete_pairs(n)
    actual = head.relative_buckets(pairs, n)
    assert set(actual.flatten().tolist()) == set(range(31))

    def scalar(start, end, token):
        distance = token - start if token < start else token - end + 1 if token >= end else 0
        magnitude = abs(distance)
        bucket = magnitude if magnitude < 8 else min(
            15, 8 + math.floor(math.log(magnitude / 8) / math.log(128 / 8) * 8 + 1e-12)
        )
        return bucket + 15 if distance > 0 else bucket

    geometry = pairs.tolist()
    for start, end in ((0, 1), (4, 9), (60, 61), (129, 130)):
        assert actual[geometry.index([start, end])].tolist() == [scalar(start, end, t) for t in range(n)]
    assert torch.equal(actual[geometry.index([20, 24]), 10:40], actual[geometry.index([30, 34]), 20:50])
    assert not actual[geometry.index([0, n])].count_nonzero()


def test_relative_attention_reads_complete_context_without_window_or_mask():
    state = invented_state()
    state["query_weight"].zero_()
    state["query_bias"].zero_()
    state[head.BIAS_KEY] = torch.linspace(-.5, .75, 31, dtype=torch.float32)
    owner = head.RelativeSpanContextAssertionNullHead(state)
    raw, _ = inputs()
    pairs = head.exact.complete_pairs(4)
    span, context, attention = owner._representations(raw, pairs)
    expected = torch.softmax(state[head.BIAS_KEY][head.relative_buckets(pairs, 4)], dim=-1)[None].expand(7, -1, -1)
    torch.testing.assert_close(attention, expected, atol=0, rtol=0)
    values = torch.nn.functional.gelu(torch.nn.functional.linear(raw, state["context_weight"], state["context_bias"]))
    torch.testing.assert_close(context, torch.matmul(expected, values), atol=0, rtol=0)
    assert span.shape == context.shape == (7, 10, 32)
    assert attention.shape == (7, 10, 4) and bool((attention > 0).all())


def test_exact_set_loss_reaches_all_13_parameters_and_relative_bias():
    owner = head.RelativeSpanContextAssertionNullHead(invented_state()).set_learning(True)
    raw, evidence = inputs()
    originals = raw.clone(), evidence.clone()
    bank = owner(raw, evidence)
    gold_sets = ((), ((0, 1, 0),), ((0, 1, 0), (2, 4, 2)))
    loss = sum(head.negative_log_likelihood(bank, gold, reduction="sum") for gold in gold_sets) / 3
    loss.backward()
    assert len(tuple(owner.parameters())) == 13
    assert all(p.grad is not None and bool(p.grad.isfinite().all()) for p in owner.parameters())
    reachable = torch.unique(head.relative_buckets(head.exact.complete_pairs(4), 4))
    assert int((owner.relative_distance_bias.grad[reachable].abs() > 1e-12).sum()) >= 2
    assert raw.grad is evidence.grad is None
    assert torch.equal(raw, originals[0]) and torch.equal(evidence, originals[1])
    owner.set_learning(False)


def test_supplied_states_two_owners_have_private_storage_and_keep_global_rng():
    state = invented_state()
    expected = head.state_digest(state)
    rng = torch.get_rng_state().clone()
    first = head.RelativeSpanContextAssertionNullHead(state)
    second = head.RelativeSpanContextAssertionNullHead(state)
    assert first.validate() == second.validate() == expected
    inventories = [state, dict(first.state_dict()), dict(second.state_dict())]
    pointers = [value.untyped_storage().data_ptr() for inventory in inventories for value in inventory.values()]
    assert len(pointers) == len(set(pointers)) == 39
    assert torch.equal(rng, torch.get_rng_state())
    state["span_bias"].add_(1)
    assert first.validate() == second.validate() == expected
    with torch.no_grad():
        first.span_bias[0].add_(.1)
    with pytest.raises(ValueError):
        first.validate()


def test_uncapped_exact_decoder_retains_four_nonoverlapping_events():
    raw, evidence = inputs()
    bank = head.RelativeSpanContextAssertionNullHead(invented_state())(raw, evidence)
    event = torch.full_like(bank.event_logits, -100.)
    null = torch.zeros_like(event)
    for index, (start, end) in enumerate(bank.pair_indices.tolist()):
        if end - start == 1:
            event[index, 0] = 20.
    scores = head.baseline.SpanContextEventNullBank(
        bank.energy_version, bank.type_logp, event, null, bank.pair_indices, bank.unit_count
    )
    assert len(head.predict(scores)["spans"]) == 4


@pytest.mark.parametrize("defect", ("missing", "extra", "shape", "nan", "alias"))
def test_complete_state_validation_rejects_missing_extra_bad_or_aliased_tensors(defect):
    state = invented_state()
    if defect == "missing":
        state.pop(head.BIAS_KEY)
    elif defect == "extra":
        state["unlicensed"] = torch.zeros(1)
    elif defect == "shape":
        state[head.BIAS_KEY] = torch.zeros(30)
    elif defect == "nan":
        state[head.BIAS_KEY][0] = float("nan")
    else:
        name = max(state, key=lambda key: state[key].numel())
        state[head.BIAS_KEY] = state[name].reshape(-1)[:31]
    with pytest.raises(ValueError):
        head.RelativeSpanContextAssertionNullHead(state)
