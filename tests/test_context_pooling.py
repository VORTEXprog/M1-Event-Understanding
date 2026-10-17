"""Full-context arithmetic on invented numerical inputs only."""
from dataclasses import replace
import math

import pytest
import torch
from torch.nn import functional as F

from m1 import full_context_head_candidate_pooling_v1 as pool
from m1.joint_argument_span_score_v1 import enumerate_unit_pairs


@pytest.fixture(autouse=True)
def cpu2(monkeypatch):
    previous = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    yield
    torch.set_num_threads(previous[0]); torch.use_deterministic_algorithms(previous[1])


def invented():
    x = (torch.arange(60, dtype=torch.float32).reshape(3, 5, 4)-23)/17
    x[0] = torch.tensor([[1., 0., .1, .2], [77., 88., 99., 66.],
        [0., 1., .2, -.3], [-1., .3, .4, .8], [.2, -.8, 1., -.2]])
    tm = torch.tensor([[1, 0, 1, 1, 1], [1, 1, 0, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    em = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    heads = torch.tensor([[[0, 1], [2, 3], [3, 5]], [[1, 2], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0]]])
    _, pm = enumerate_unit_pairs(tm)
    v = (torch.arange(3*3*pm.shape[1]*3, dtype=torch.float32).reshape(3, 3, -1, 3)-57)/83
    return pool.FullContextPoolingInputs(x, tm, em, heads, v, pm)


def learned(chunk=128):
    net = pool.FullContextHeadCandidatePooling(4, 3, 3, chunk)
    with torch.no_grad(): net.output_weight.copy_(torch.eye(3))
    return net


def singleton(result, unit=2, row=0):
    index = ((result.pair_indices[row] == torch.tensor([unit, unit])).all(-1)
        & result.pair_mask[row]).nonzero().flatten()
    assert len(index) == 1
    return int(index[0])


def reference(net, data, result):
    expected = torch.zeros_like(result.projected_correction)
    for b in range(len(data.unit_states)):
        context = data.unit_states[b, data.token_mask[b]]
        keys, values = F.linear(context, net.key_weight), F.linear(context, net.value_weight)
        for e in data.event_mask[b].nonzero().flatten().tolist():
            left, right = data.event_unit_spans[b, e].tolist()
            head = data.unit_states[b, left:right][data.token_mask[b, left:right]].mean(0)
            for p in data.pair_mask[b].nonzero().flatten().tolist():
                q = F.linear(head, net.head_weight) + F.linear(data.candidate_vectors[b, e, p], net.candidate_weight)
                alpha = ((keys*q).sum(-1)/math.sqrt(net.attention_dimension)).softmax(-1)
                expected[b, e, p] = F.linear((alpha[:, None]*values).sum(0), net.output_weight)
    return expected


def test_complete_context_reference_full_spans_holes_e0_e1_e3_and_shared_candidates():
    data, net = invented(), learned()
    result = net(data)
    assert result.pair_mask.sum(-1).tolist() == [10, 6, 1]
    assert result.projected_correction.shape == (3, 3, 10, 3)
    torch.testing.assert_close(result.projected_correction, reference(net, data, result), rtol=2e-6, atol=2e-6)
    valid = data.event_mask[..., None] & data.pair_mask[:, None]
    assert not bool(result.projected_correction[~valid].count_nonzero())
    p = singleton(result)
    assert bool(data.event_mask[0].all()) and bool(result.pair_mask[0, p])
    assert not torch.equal(result.projected_correction[0, 0, p], result.projected_correction[0, 1, p])


def test_singleton_depends_on_own_head_with_candidate_and_sentence_fixed():
    data, net = invented(), learned()
    before = net(data); p = singleton(before)
    heads = data.event_unit_spans.clone(); heads[0, 0] = torch.tensor([3, 4])
    after = net(replace(data, event_unit_spans=heads))
    assert not torch.allclose(before.projected_correction[0, 0, p], after.projected_correction[0, 0, p])
    assert torch.equal(before.projected_correction[0, 1:], after.projected_correction[0, 1:])


def test_singleton_depends_on_outside_span_context_with_own_head_fixed():
    data, net = invented(), learned()
    before = net(data); p = singleton(before)
    x = data.unit_states.clone(); x[0, 4, :3] = torch.tensor([2., -2., 3.])
    after = net(replace(data, unit_states=x))
    assert torch.equal(x[0, 0], data.unit_states[0, 0]) and torch.equal(x[0, 2], data.unit_states[0, 2])
    assert not torch.allclose(before.projected_correction[0, 0, p], after.projected_correction[0, 0, p])


def test_candidate_query_dependency_at_fixed_head_and_normalized_product_identity():
    data, net = invented(), learned()
    before = net(data); p = singleton(before)
    v = data.candidate_vectors.clone(); v[0, 0, p] += torch.tensor([2., -3., 1.])
    after = net(replace(data, candidate_vectors=v))
    assert not torch.allclose(before.projected_correction[0, 0, p], after.projected_correction[0, 0, p])
    x = data.unit_states[0, data.token_mask[0]]
    k = F.linear(x, net.key_weight)
    a = (k*F.linear(data.unit_states[0, 0], net.head_weight)).sum(-1)/math.sqrt(3)
    b = (k*F.linear(data.candidate_vectors[0, 0, p], net.candidate_weight)).sum(-1)/math.sqrt(3)
    product = a.softmax(-1)*b.softmax(-1)
    torch.testing.assert_close((a+b).softmax(-1), product/product.sum(), rtol=2e-6, atol=2e-6)


def test_literal_warm_zero_correction_no_rng_and_exact_initial_gradient_pattern():
    rng = torch.random.get_rng_state().clone()
    production = pool.FullContextHeadCandidatePooling()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert len(production.state_dict()) == 5 and sum(p.numel() for p in production.parameters()) == pool.PARAMETERS
    assert torch.equal(production.head_weight, torch.eye(32, 128))
    assert torch.equal(production.candidate_weight, torch.eye(32))
    net, data = pool.FullContextHeadCandidatePooling(4, 3, 3), invented()
    result = net(data); p = singleton(result)
    assert not bool(result.projected_correction.count_nonzero())
    assert torch.equal(data.candidate_vectors+result.projected_correction, data.candidate_vectors)
    result.projected_correction[0, 0, p, 0].backward()
    for key, weight in net.named_parameters():
        assert weight.grad is not None and bool(torch.isfinite(weight.grad).all())
        assert bool(weight.grad.count_nonzero()) == (key == "output_weight")


def test_general_state_gradients_reach_five_weights_ordered_states_and_native_vectors():
    data, net = invented(), learned()
    data = replace(data, unit_states=data.unit_states.clone().requires_grad_(),
        candidate_vectors=data.candidate_vectors.clone().requires_grad_())
    result = net(data); p = singleton(result)
    result.projected_correction[0, 0, p].square().sum().backward()
    for value in (*net.parameters(), data.unit_states, data.candidate_vectors):
        assert value.grad is not None and bool(torch.isfinite(value.grad).all()) and bool(value.grad.count_nonzero())


def test_query_chunks_two_versus_all_preserve_values_and_gradients():
    data, small, large = invented(), learned(2), learned(128)
    first, second = small(data), large(data)
    torch.testing.assert_close(first.projected_correction, second.projected_correction, rtol=2e-6, atol=2e-6)
    first.projected_correction.square().sum().backward(); second.projected_correction.square().sum().backward()
    for a, b in zip(small.parameters(), large.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-6, atol=3e-6)


def test_event_permutation_holes_and_e0_leave_masks_inputs_and_padding_unchanged():
    data, net = invented(), learned()
    before = tuple(v.clone() for v in vars(data).values()); original = net(data)
    permutation = torch.tensor([2, 0, 1])
    permuted = replace(data, event_mask=data.event_mask[:, permutation],
        event_unit_spans=data.event_unit_spans[:, permutation], candidate_vectors=data.candidate_vectors[:, permutation])
    torch.testing.assert_close(net(permuted).projected_correction, original.projected_correction[:, permutation])
    x = data.unit_states.clone(); x[~data.token_mask] = -999
    assert torch.equal(net(replace(data, unit_states=x)).projected_correction, original.projected_correction)
    empty = replace(data, event_mask=torch.zeros_like(data.event_mask),
        event_unit_spans=torch.zeros_like(data.event_unit_spans))
    result = net(empty)
    assert not bool(result.projected_correction.count_nonzero())
    result.projected_correction.sum().backward()
    assert all(p.grad is not None and not bool(p.grad.count_nonzero()) for p in net.parameters())
    net.zero_grad(set_to_none=True)
    empty_axis = replace(data, event_mask=data.event_mask[:, :0],
        event_unit_spans=data.event_unit_spans[:, :0], candidate_vectors=data.candidate_vectors[:, :0])
    axis_result = net(empty_axis)
    assert axis_result.projected_correction.shape == (3, 0, 10, 3)
    axis_result.projected_correction.sum().backward()
    assert all(p.grad is not None and not bool(p.grad.count_nonzero()) for p in net.parameters())
    for old, current in zip(before, vars(data).values()): assert torch.equal(old, current)
    assert torch.equal(result.token_mask, data.token_mask) and torch.equal(result.pair_mask, data.pair_mask)


def test_exact_supplied_restore_independent_storage_no_rng_or_input_state_mutation():
    data, state = invented(), pool.export_state(learned())
    saved = {k: v.clone() for k, v in state.items()}; rng = torch.random.get_rng_state().clone()
    first = pool.model_from_state(state, hidden_dimension=4, span_dimension=3, attention_dimension=3)
    second = pool.model_from_state(state, hidden_dimension=4, span_dimension=3, attention_dimension=3)
    assert torch.equal(rng, torch.random.get_rng_state()) and not first.training
    assert not any(p.requires_grad for p in first.parameters())
    assert torch.equal(first(data).projected_correction, second(data).projected_correction)
    for key in pool.STATE_KEYS:
        assert torch.equal(state[key], saved[key])
        assert len({state[key].data_ptr(), first.state_dict()[key].data_ptr(), second.state_dict()[key].data_ptr()}) == 3


def test_pruned_pair_inventory_invalid_head_extra_state_and_chunk_limit_rejected():
    data, net = invented(), learned()
    pm = data.pair_mask.clone(); pm[0, -1] = False
    heads = data.event_unit_spans.clone(); heads[0, 0] = torch.tensor([1, 3])
    for bad in (replace(data, pair_mask=pm), replace(data, event_unit_spans=heads),
                replace(data, candidate_vectors=data.candidate_vectors[:, :, :-1])):
        with pytest.raises(ValueError): net(bad)
    state = pool.export_state(net); state["unlicensed_weight"] = torch.ones(1)
    with pytest.raises(ValueError): pool.model_from_state(state, hidden_dimension=4, span_dimension=3, attention_dimension=3)
    with pytest.raises(ValueError): pool.FullContextHeadCandidatePooling(query_chunk_size=129)
