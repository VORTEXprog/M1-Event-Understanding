"""Native TYPE readout on invented states and complete feature matrices only."""
from dataclasses import replace
import hashlib
import inspect

import pytest
import torch

from m1.canonical_units import segment_text
from m1 import semantic_prefix_conditional_identity_contract_v1 as contract
from m1 import semantic_prefix_conditional_identity_native_v1 as native


def supplied_state(architecture):
    values = {key: torch.linspace(-.07, .08, __import__('math').prod(shape)).reshape(shape).contiguous()
        for key, shape in contract.SHAPES[architecture].items()}
    values['unit_weight'].zero_()
    values['unit_weight'][:, :16] = torch.eye(16)*.03
    return values


def inputs(texts=('Participant_A provided object. Tail stays.', 'Участник_Б сдвинул щит. e\u0301 Z', 'One')):
    rows = tuple(contract.RawRow(f'invented/native-prefix/{i}', text, 'ru' if i == 1 else 'en',
        tuple((u.char_start, u.char_end) for u in segment_text(text))) for i, text in enumerate(texts))
    raw_sha = contract.raw_sample_digest(rows); records = []
    for index, row in enumerate(rows):
        for k in range(7):
            x = torch.zeros((len(row.units), 2048), dtype=torch.float32)
            x[:, :16] = torch.arange(len(row.units)*16, dtype=torch.float32).reshape(-1, 16)*.03-.3
            x[:, 8+k] += .07*(k+1)+index*.02
            item = contract.candidate_item_id(raw_sha, row.sentence_id, k)
            records.append(contract.PrefixUnitRecord(item, row.sentence_id, k,
                contract.FORWARD_BASIS_SHA256, row.units, item,
                hashlib.sha256(x.numpy().tobytes()).hexdigest(), x))
    records = tuple(records)
    return native.authenticate_inputs(rows=rows, records=records, raw_sample_sha256=raw_sha,
        input_sha256=contract.input_commitment(rows, records), fixture_only_not_admissible=True)


def owner(architecture):
    return native.owner_from_state(supplied_state(architecture), architecture,
        fixture_only_not_admissible=True)


def test_static_basis_per_call_identity_and_metadata_only_sink_commitment():
    one, two = inputs(), inputs(('A different arbitrary raw turn.',))
    assert one.forward_basis_sha256 == two.forward_basis_sha256 == contract.FORWARD_BASIS_SHA256
    assert one.raw_sample_sha256 != two.raw_sample_sha256
    assert one.input_sha256 != two.input_sha256
    assert contract.readout_basis('shared_bigru16') != contract.readout_basis('mean_context')
    assert contract.FORWARD_RECIPE['prefixes'] == tuple(contract.meaningful_prefix(k) for k in range(7))
    assert contract.input_commitment(one.rows, tuple(replace(r, fp32_units=None) for r in one.records)) == one.input_sha256
    assert contract.candidate_item_id(one.raw_sample_sha256, 'same-id', 0) != contract.candidate_item_id(
        two.raw_sample_sha256, 'same-id', 0)




@pytest.mark.parametrize('architecture', contract.ARCHITECTURES)
def test_stable_native_clone_frozen_export_restore_no_rng(architecture):
    supplied = supplied_state(architecture); before_rng = torch.get_rng_state().clone()
    first = native.owner_from_state(supplied, architecture, fixture_only_not_admissible=True)
    assert type(first) is native.SemanticPrefixConditionalIdentityOwner
    assert torch.equal(before_rng, torch.get_rng_state())
    assert not first.training and all(not p.requires_grad and p.grad is None for p in first.parameters())
    assert len(tuple(first.parameters())) == (6 if architecture == 'mean_context' else 14)
    assert sum(p.numel() for p in first.parameters()) == contract.PARAMETERS[architecture]
    exported, metadata = native.export_cpu_fp32_state(first)
    second = native.owner_from_state(exported, architecture, fixture_only_not_admissible=True)
    for group in (supplied, dict(first.named_parameters()), exported, dict(second.named_parameters())):
        assert all(torch.equal(value, supplied[key]) for key, value in group.items())
    groups = [{v.untyped_storage().data_ptr() for v in group.values()} for group in
        (supplied, dict(first.named_parameters()), exported, dict(second.named_parameters()))]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i+1:])
    assert metadata['state_sha256'] == second.validate_frozen() == first.validate_frozen()
    assert metadata['old_native75_owner'] is False
    bundle = inputs()
    assert torch.equal(native.score_complete_dense(bundle, first).scores,
        native.score_complete_dense(bundle, second).scores)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert torch.equal(native.score_complete_dense(inputs(('Another turn.',)), first).scores,
        native.score_complete_dense(inputs(('Another turn.',)), second).scores)




@pytest.mark.parametrize('architecture', contract.ARCHITECTURES)
def test_exact_variable_detected_heads_whole_char_and_e0(architecture):
    bundle = inputs(); bank = native.score_complete_dense(bundle, owner(architecture))
    row = bundle.rows[1]; h0, h1, h2 = row.units[:3]
    heads = (contract.DetectedCharHead(row.sentence_id, h2[0], h2[1]),
        contract.DetectedCharHead(row.sentence_id, h0[0], h1[1]),
        contract.DetectedCharHead(row.sentence_id, h0[0], h0[1]),
        contract.DetectedCharHead(bundle.rows[2].sentence_id, *bundle.rows[2].units[0]))
    out = native.detected_head_readout(bank, heads)
    assert out.heads == heads and out.scores.shape == (4, 7) and out.type_indices.shape == (4,)
    assert torch.equal(out.scores[3], bank.scores[bank.row_pair_offsets[2]])
    assert torch.equal(out.scores[1], bank.scores[bank.row_pair_offsets[1]+1])
    assert out.type_names == tuple(contract.TYPES[k] for k in out.type_indices.tolist())
    empty = native.detected_head_readout(bank, ())
    assert empty.heads == () and empty.scores.shape == (0, 7) and empty.type_names == ()
    assert empty.type_indices.dtype == torch.int64 and empty.type_indices.numel() == 0
    assert set(inspect.signature(native.detected_head_readout).parameters) == {'bank', 'heads'}
    assert set(inspect.signature(native.score_complete_dense).parameters) == {'inputs', 'owner'}


def test_exact_ties_resolve_first_native_type_and_scores_are_independent_clones():
    bundle = inputs(); state = {k: torch.zeros(shape) for k, shape in contract.SHAPES['mean_context'].items()}
    model = native.owner_from_state(state, 'mean_context', fixture_only_not_admissible=True)
    bank = native.score_complete_dense(bundle, model); row = bundle.rows[0]
    head = contract.DetectedCharHead(row.sentence_id, *row.units[0])
    out = native.detected_head_readout(bank, (head,))
    assert out.type_indices.tolist() == [0] and out.type_names == (contract.TYPES[0],)
    out.scores.add_(1)
    assert not bool(bank.scores.any())


@pytest.mark.parametrize('fault', ['partial', 'old_basis', 'reorder', 'alias', 'wrong_id', 'payload'])
def test_input_geometry_static_basis_and_candidate_custody_fail_closed(fault):
    bundle = inputs(); rows, records = bundle.rows, list(bundle.records)
    if fault == 'partial':
        rows = (replace(rows[0], units=rows[0].units[:-1]), *rows[1:])
    elif fault == 'old_basis': records[0] = replace(records[0], forward_basis_sha256='e'*64)
    elif fault == 'reorder': records[:2] = reversed(records[:2])
    elif fault == 'alias': records[1] = replace(records[1], fp32_units=records[0].fp32_units,
        fp32_units_sha256=records[0].fp32_units_sha256)
    elif fault == 'wrong_id': records[0] = replace(records[0], item_id='f'*64)
    elif fault == 'payload': records[0] = replace(records[0], fp32_units_sha256='f'*64)
    records = tuple(records)
    with pytest.raises(ValueError):
        native.authenticate_inputs(rows=rows, records=records,
            raw_sample_sha256=contract.raw_sample_digest(rows),
            input_sha256=contract.input_commitment(rows, records), fixture_only_not_admissible=True)


@pytest.mark.parametrize('fault', ['state', 'pointer', 'train', 'grad', 'raw', 'raw_data'])
def test_frozen_state_pointer_training_and_raw_mutation_fail_closed(fault):
    bundle = inputs(); model = owner('shared_bigru16')
    if fault == 'state': model.type_bias.data.add_(.1)  # bypasses version; byte digest must still reject
    elif fault == 'pointer': model.type_bias = torch.nn.Parameter(model.type_bias.detach().clone(), requires_grad=False)
    elif fault == 'train': model.train()
    elif fault == 'grad': model.type_bias.requires_grad_(True)
    elif fault == 'raw': bundle.records[0].fp32_units.add_(.1)
    elif fault == 'raw_data': bundle.records[0].fp32_units.data.add_(.1)
    with pytest.raises(ValueError): native.score_complete_dense(bundle, model)


@pytest.mark.parametrize('head', [contract.DetectedCharHead('invented/absent', 0, 1),
    contract.DetectedCharHead('invented/native-prefix/0', 0, 2),
    contract.DetectedCharHead('invented/native-prefix/0', 4, 1),
    contract.DetectedCharHead('invented/native-prefix/0', False, 4)])
def test_invalid_partial_reversed_unknown_and_bool_heads_rejected(head):
    bank = native.score_complete_dense(inputs(), owner('mean_context'))
    with pytest.raises(ValueError): native.detected_head_readout(bank, (head,))


def test_duplicate_detected_head_event_key_rejected_without_overlap_rule():
    bundle = inputs(); bank = native.score_complete_dense(bundle, owner('mean_context'))
    row = bundle.rows[0]; first, second = row.units[:2]
    one = contract.DetectedCharHead(row.sentence_id, *first)
    with pytest.raises(ValueError, match='unique'):
        native.detected_head_readout(bank, (one, one))
    wider = contract.DetectedCharHead(row.sentence_id, first[0], second[1])
    assert native.detected_head_readout(bank, (wider, one)).heads == (wider, one)


def test_closed_state_rejects_extra_missing_nonfinite_and_alias():
    state = supplied_state('mean_context')
    for broken in ({**state, 'old_native75': torch.zeros(1)}, dict(list(state.items())[1:]),
                   {**state, 'type_bias': torch.full((7,), float('nan'))}):
        with pytest.raises(ValueError): native.owner_from_state(broken, 'mean_context')
    bad = supplied_state('shared_bigru16'); bad['recurrent_bias_hh'] = bad['recurrent_bias_ih']
    with pytest.raises(ValueError): native.owner_from_state(bad, 'shared_bigru16')


@pytest.mark.parametrize('architecture', contract.ARCHITECTURES)
def test_one_candidate_feature_view_changes_only_its_own_score_column_and_row(architecture):
    before_inputs = inputs()
    model = owner(architecture)
    before = native.score_complete_dense(before_inputs, model)
    records = list(before_inputs.records)
    chosen = records[3]
    values = chosen.fp32_units.clone()
    values[:, 3] += torch.arange(len(values), dtype=torch.float32) * .1 + 2.
    records[3] = replace(chosen, fp32_units=values,
        fp32_units_sha256=hashlib.sha256(values.numpy().tobytes()).hexdigest())
    records = tuple(records)
    changed = native.authenticate_inputs(rows=before_inputs.rows, records=records,
        raw_sample_sha256=before_inputs.raw_sample_sha256,
        input_sha256=contract.input_commitment(before_inputs.rows, records), fixture_only_not_admissible=True)
    after = native.score_complete_dense(changed, model)
    end = before.row_pair_offsets[1]
    untouched_columns = [index for index in range(7) if index != 3]
    assert torch.equal(before.scores[:, untouched_columns].view(torch.uint8), after.scores[:, untouched_columns].view(torch.uint8))
    assert torch.equal(before.scores[end:].view(torch.uint8), after.scores[end:].view(torch.uint8))
    assert not torch.equal(before.scores[:end, 3], after.scores[:end, 3])
