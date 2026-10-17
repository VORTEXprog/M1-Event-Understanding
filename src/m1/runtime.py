"""Feature-first current-turn inference with complete variable-event outputs.

The runtime consumes complete unprimed and seven candidate-conditioned feature
banks. A provider is responsible for producing the declared feature basis;
accepting supplied tensors is not a claim that their origin was verified.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from hashlib import sha256
import json
from typing import Protocol
import torch

from .canonical_units import segment_text
from . import semantic_prefix_conditional_identity_native_v1 as native
from . import semantic_prefix_conditional_identity_contract_v1 as contract
from . import frozen_conditional_type_intervals_v1 as intervals
from . import span_relative_context_assertion_null_v1 as local
from . import event_sentence_recurrent_identity_ensemble_v1 as original
from . import three_member_conditioned_polarity_heads_v1 as head
from . import event_conditioned_pair_owner_v1 as pair
from . import full_context_parallel_binding_owner_v1 as full
from . import event_conditioned_pair_saved_port_v2 as saved
from . import event_anchor_argument_compatibility_v1 as anchors
from .inference_v2 import ObservedSentence, observed_inputs
from .polarity_decision_v1 import decide_polarity
from .tensor_state import state_dict_sha256

COMPONENT_NAMES = ('type14', 'localizer', 'original66', 'heads', 'binding')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _token(value):
    if isinstance(value, torch.Tensor):
        return (id(value), value.untyped_storage().data_ptr(), value._version,
                str(value.dtype), tuple(value.shape), sha256(value.detach().contiguous().numpy().tobytes()).hexdigest())
    if type(value) is dict:
        return tuple((key, _token(item)) for key, item in value.items())
    if type(value) in (tuple, list):
        return tuple(_token(item) for item in value)
    if hasattr(value, '__dataclass_fields__'):
        return tuple((f.name, _token(getattr(value, f.name))) for f in fields(value)
                     if not f.name.startswith('_'))
    return value


def _fp32(value, shape):
    require(type(value) is torch.Tensor and tuple(value.shape) == tuple(shape)
        and value.device.type == 'cpu' and value.dtype == torch.float32 and value.is_contiguous()
        and not value.requires_grad and not value.is_inference() and bool(value.isfinite().all()),
        'ordinary detached finite contiguous CPU FP32 features required')


@dataclass(frozen=True)
class RawText:
    sentence_id: str
    text: str
    language: str

    def validate(self):
        require(type(self) is RawText and set(vars(self)) == {'sentence_id', 'text', 'language'}
            and type(self.sentence_id) is str and bool(self.sentence_id)
            and type(self.text) is str and bool(self.text) and self.language in ('en', 'ru'),
            'closed sentence_id/text/language input required')
        self.text.encode('utf8')
        return segment_text(self.text)


@dataclass(frozen=True)
class FeatureBatch:
    rows: tuple[RawText, ...]
    unprimed: tuple[torch.Tensor, ...]
    prefixes: tuple[tuple[torch.Tensor, ...], ...]
    provider_metadata: dict = field(default_factory=dict)
    _commitment: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._validate_geometry()
        object.__setattr__(self, '_commitment', _token((self.rows, self.unprimed, self.prefixes, self.provider_metadata)))

    def _validate_geometry(self):
        require(type(self.rows) is tuple and self.rows and type(self.unprimed) is tuple
            and type(self.prefixes) is tuple and len(self.rows) == len(self.unprimed) == len(self.prefixes)
            and all(type(row) is RawText for row in self.rows)
            and len({row.sentence_id for row in self.rows}) == len(self.rows),
            'complete unique ordered feature rows required')
        pointers = []
        for row, raw, prefixed in zip(self.rows, self.unprimed, self.prefixes, strict=True):
            require(type(row) is RawText, 'exact annotation-free RawText required')
            units = row.validate()
            _fp32(raw, (len(units), 2048))
            require(type(prefixed) is tuple and len(prefixed) == 7,
                    'all seven candidate-conditioned banks are mandatory even for E0')
            for value in prefixed:
                _fp32(value, (len(units), 2048))
            pointers.extend(value.untyped_storage().data_ptr() for value in (raw, *prefixed))
        require(len(set(pointers)) == len(pointers), 'raw and candidate feature storages must be disjoint')
        require(type(self.provider_metadata) is dict, 'detached provider metadata object required')
        try:
            json.dumps(self.provider_metadata, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError('provider metadata must be finite JSON values') from error

    def validate(self):
        self._validate_geometry()
        require(_token((self.rows, self.unprimed, self.prefixes, self.provider_metadata)) == self._commitment,
                'retained current text/features/provider metadata changed')
        return self


class FeatureProvider(Protocol):
    def capture(self, rows: tuple[RawText, ...]) -> FeatureBatch:
        """Capture all unprimed and row×7 banks before any event prediction."""
        ...


@dataclass(frozen=True)
class SuppliedComponents:
    type14: object
    localizer: object
    original66: object
    heads: object
    binding: object
    _commitment: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._validate_owners()
        object.__setattr__(self, '_commitment', self._owners_token())

    def _owners_token(self):
        return tuple((name, id(getattr(self, name)),
                      tuple((key, _token(value)) for key, value in getattr(self, name).named_parameters()))
                     for name in COMPONENT_NAMES)

    def _validate_owners(self):
        require(type(self.type14) is native.SemanticPrefixConditionalIdentityOwner
            and self.type14.architecture == 'shared_bigru16'
            and type(self.localizer) is local.RelativeSpanContextAssertionNullHead
            and type(self.original66) is original.SentenceRecurrentIdentityProbabilityEnsemble
            and type(self.heads) is head.ThreeMemberConditionedPolarityHeads
            and type(self.binding) in (pair.EventConditionedPairArgumentExpert, full.FullContextParallelBindingExpert),
            'complete exact TYPE14/LOC13/original66/POLREL96/native74-or83 owners required')
        self.type14.validate_frozen()
        self.localizer.validate()
        original._validate_members(self.original66)
        head.validate_model(self.heads)
        self.binding._validate_topology()
        require(self.heads.source_bindings.original_identity66_sha256
            == state_dict_sha256(dict(self.original66.state_dict())),
            'head96 conditioning state commitment differs from supplied original66')
        pointers = []
        for name in COMPONENT_NAMES:
            owner = getattr(self, name)
            require(all(not module.training and getattr(module, '_compiled_call_impl', None) is None
                and 'forward' not in vars(module)
                and not any(callable(getattr(type(module), key, None)) for key in vars(module))
                and not any(getattr(module, key, None) for key in
                    ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks'))
                for module in owner.modules()), 'frozen ordinary owners without execution hooks required')
            for value in owner.parameters():
                _fp32(value.detach(), value.shape)
                require(not value.requires_grad and value.grad is None, 'frozen gradient-free owners required')
                pointers.append(value.untyped_storage().data_ptr())
        require(len(set(pointers)) == len(pointers), 'all five owner storages must be independent')

    def validate(self):
        self._validate_owners()
        require(self._owners_token() == self._commitment, 'supplied owner identity/state changed')
        return self


@dataclass(frozen=True)
class BindingBatch:
    first_row: int
    sentence_ids: tuple
    inputs: object
    scores: object
    decisions: torch.Tensor


@dataclass(frozen=True)
class ComponentResult:
    features: FeatureBatch
    models: SuppliedComponents
    type_bank: object
    local_banks: tuple
    localization: tuple
    factors: tuple
    binding_batches: tuple[BindingBatch, ...]
    saved_binding_rows: tuple
    factor_state: dict
    provenance: dict
    _commitment: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, '_commitment', self._retained())

    def _retained(self):
        return _token((self.type_bank, self.local_banks, self.localization, self.factors,
                       self.binding_batches, self.saved_binding_rows, self.factor_state, self.provenance))

    def validate(self):
        self.features.validate()
        self.models.validate()
        intervals.validate_intervals(self.type_bank)
        require(self._retained() == self._commitment, 'retained complete inference result changed')
        return self


def _prefix_inputs(features, owner):
    rows = tuple(contract.RawRow(row.sentence_id, row.text, row.language,
        tuple((unit.char_start, unit.char_end) for unit in segment_text(row.text))) for row in features.rows)
    sample = contract.raw_sample_digest(rows)
    records = []
    for row, values in zip(rows, features.prefixes, strict=True):
        for candidate, value in enumerate(values):
            item = contract.candidate_item_id(sample, row.sentence_id, candidate)
            records.append(contract.PrefixUnitRecord(item, row.sentence_id, candidate,
                contract.FORWARD_BASIS_SHA256, row.units, item, native._fp32_sha(value), value))
    records = tuple(records)
    return native.authenticate_inputs(rows=rows, records=records, raw_sample_sha256=sample,
        input_sha256=contract.input_commitment(rows, records),
        fixture_only_not_admissible=owner.fixture_only_not_admissible)


def _anchored(rows, semantic):
    geometry = torch.zeros((*semantic.event_mask.shape, 2), dtype=torch.int64)
    for b, row in enumerate(rows):
        units = segment_text(row.text)
        for e, (start, stop) in enumerate(row.event_spans):
            selected = [u.unit_index for u in units if u.char_start < stop and u.char_end > start]
            require(selected and units[selected[0]].char_start == start
                and units[selected[-1]].char_end == stop, 'predicted HEAD must cover whole canonical units')
            geometry[b, e] = torch.tensor((selected[0], selected[-1] + 1), dtype=torch.int64)
    result = anchors.AnchoredArgumentInputs(semantic, geometry)
    result.validate()
    return result


def _factor_state(owner):
    source = owner.parent if type(owner) is pair.EventConditionedPairArgumentExpert else owner.native74.parent
    return {key: value.detach().clone().contiguous() for key, value in {
        'null_embedding': source.null_embedding, 'factor_left.weight': source.factor_left.weight,
        'factor_right.weight': source.factor_right.weight}.items()}


def predict_features(features, models, *, budget_check=None):
    require(type(features) is FeatureBatch and type(models) is SuppliedComponents
        and torch.get_num_threads() == 2 and torch.get_default_dtype() == torch.float32
        and torch.get_default_device().type == 'cpu'
        and (budget_check is None or callable(budget_check)), 'complete supplied features/owners and ordinary CPU2 FP32 required')
    guard = budget_check or (lambda: None)
    features.validate(); models.validate()
    owner_pointers = {p.untyped_storage().data_ptr() for name in COMPONENT_NAMES for p in getattr(models, name).parameters()}
    feature_pointers = {v.untyped_storage().data_ptr() for raw, prefixed in zip(features.unprimed, features.prefixes, strict=True)
                        for v in (raw, *prefixed)}
    require(not owner_pointers.intersection(feature_pointers), 'feature/state alias rejected')
    rng = torch.random.get_rng_state().clone()
    banks, locations, observed = [], [], []
    with torch.inference_mode(False), torch.no_grad():
        guard(); inputs = _prefix_inputs(features, models.type14)
        aligned = intervals.align_complete_bank(native.score_complete_dense(inputs, models.type14)); guard()
        for row, raw, values, begin, end in zip(features.rows, features.unprimed, features.prefixes,
                aligned.row_pair_offsets[:-1], aligned.row_pair_offsets[1:], strict=True):
            bank = models.localizer(torch.stack(values).contiguous(), aligned.logits[begin:end])
            prediction = local.predict(bank)
            units = segment_text(row.text)
            events = tuple({'event_index': i, 'char_start': units[start].char_start,
                'char_end': units[stop - 1].char_end, 'TYPE': contract.TYPES[kind]}
                for i, (start, stop, kind) in enumerate(prediction['spans']))
            banks.append(bank); locations.append({'sentence_id': row.sentence_id, 'events': events,
                                                  'map_score': prediction['map_score']})
            observed.append(ObservedSentence(row.sentence_id, row.text, row.language, raw,
                            tuple((e['char_start'], e['char_end']) for e in events)))
            guard()
        batches, saved_rows, factors = [], [], []
        factor_state = _factor_state(models.binding)
        for first in range(0, len(observed), 8):
            rows = tuple(observed[first:first + 8]); guard()
            semantic = observed_inputs(rows, 2048)
            scores7 = models.original66(semantic).detach().clone().contiguous()
            head_output = models.heads(semantic, scores7)
            polarities = decide_polarity(head_output.polarity_logits, semantic.event_mask)
            relations = head_output.relation_logits.argmax(-1)
            anchored = _anchored(rows, semantic)
            potentials = models.binding(anchored)
            potentials.validate()
            decisions = pair.parent.native.decode_beam8(potentials).detach().clone().contiguous()
            batches.append(BindingBatch(first, tuple(r.sentence_id for r in rows), anchored, potentials, decisions))
            null = semantic.token_states.shape[1]
            for b, row in enumerate(rows):
                units, events = segment_text(row.text), []
                saved_rows.append(saved.pack_row(potentials, b, unit_width=len(units), factor_state=factor_state))
                for e, (start, stop) in enumerate(row.event_spans):
                    roles = []
                    for role, (left, right) in enumerate(decisions[b, e].tolist()):
                        if left == right == null:
                            continue
                        require(0 <= left <= right < len(units), 'original-unit mention or native NULL required')
                        a, z = units[left].char_start, units[right].char_end
                        roles.append({'role_index': role, 'char_start': a, 'char_end': z, 'surface': row.text[a:z]})
                    events.append({'event_index': e, 'char_start': start, 'char_end': stop,
                        'TYPE': locations[first + b]['events'][e]['TYPE'], 'polarity_id': int(polarities[b, e]), 'roles': roles})
                edges = [{'source_event_index': e, 'target_event_index': f, 'relation_id': int(relations[b, e, f])}
                    for e in range(len(events)) for f in range(len(events)) if e != f and int(relations[b, e, f]) != 0]
                factors.append({'sentence_id': row.sentence_id, 'events': events, 'relations': edges,
                                'legacy_dense_capacity_exceeded': len(events) > 2})
            guard()
    require(torch.equal(rng, torch.random.get_rng_state()), 'inference changed global CPU RNG')
    features.validate(); models.validate()
    result = ComponentResult(features, models, aligned, tuple(banks), tuple(locations), tuple(factors),
        tuple(batches), tuple(saved_rows), factor_state,
        {'all_rows_all7_before_LOC': True, 'event_count_cap': None, 'Gold_inputs': 0,
         'owner_tensors': {name: len(getattr(models, name).state_dict()) for name in COMPONENT_NAMES},
         'feature_origin_verified': False, 'runtime_admission': False, 'full_M1_qualification_claimed': False})
    replay_saved_binding(result, result.saved_binding_rows)
    guard()
    return result.validate()


def replay_saved_binding(result, reopened_rows):
    require(type(result) is ComponentResult and type(reopened_rows) is tuple
        and len(reopened_rows) == len(result.features.rows), 'complete per-row saved inventory required')
    result.validate()
    index = 0
    for batch in result.binding_batches:
        for b, sid in enumerate(batch.sentence_ids):
            manifest, bank = reopened_rows[index]
            expected_manifest, expected_bank = result.saved_binding_rows[index]
            require(manifest == expected_manifest and type(bank) is dict and set(bank) == saved.BANK_KEYS
                and all(saved._bits(value, bank[key]) for key, value in expected_bank.items()),
                'complete reopened manifest/nine-array bits differ')
            restored = saved.reconstruct_row(manifest, bank, factor_state=result.factor_state)
            expected = batch.decisions[b].clone()
            expected[(expected == batch.scores.token_mask.shape[1]).all(-1)] = manifest['unit_width']
            require(torch.equal(pair.parent.native.decode_beam8(restored)[0], expected),
                    'saved native joint decoder replay differs')
            index += 1
    return {'rows': index, 'events': sum(len(row['events']) for row in result.factors),
            'all9_actual3_native_replay_PASS': True, 'qualification_claimed': False}


def predict_raw(rows, *, models, provider, budget_check=None):
    require(type(rows) is tuple and rows and all(type(row) is RawText for row in rows)
        and callable(getattr(provider, 'capture', None)), 'annotation-free RAW tuple and feature provider required')
    require(type(models) is SuppliedComponents and torch.get_num_threads() == 2
        and torch.get_default_dtype() == torch.float32 and torch.get_default_device().type == 'cpu'
        and (budget_check is None or callable(budget_check)), 'admitted supplied owners and ordinary CPU2 FP32 required before capture')
    models.validate()
    for row in rows:
        row.validate()
    before = _token(rows)
    features = provider.capture(rows)
    require(type(features) is FeatureBatch and features.rows == rows and _token(rows) == before,
            'provider must return every original RAW row in order')
    return predict_features(features, models, budget_check=budget_check)


def compose_components(raw_rows, prefix_inputs, models, *, budget_check=None):
    """Supplied-feature adapter retaining the original composition call shape.

    RAW rows expose sentence_id/text/language/unit_states. Prefix inputs use
    the native complete seven-candidate DTO. The returned public result carries
    complete factors and saved arrays, without historical evidence envelopes.
    """
    require(type(raw_rows) is tuple and type(prefix_inputs) is native.PrefixReadoutInputs,
            'complete supplied RAW tuple and native prefix inputs required')
    native._unchanged(prefix_inputs)
    require(len(raw_rows) == len(prefix_inputs.rows), 'complete RAW/prefix row join required')
    records = {(record.sentence_id, record.candidate): record for record in prefix_inputs.records}
    rows, raw, prefixes = [], [], []
    for supplied, source in zip(raw_rows, prefix_inputs.rows, strict=True):
        require((supplied.sentence_id, supplied.text, supplied.language)
                == (source.sentence_id, source.text, source.language), 'exact original RAW/prefix join required')
        rows.append(RawText(supplied.sentence_id, supplied.text, supplied.language))
        raw.append(supplied.unit_states)
        prefixes.append(tuple(records[(source.sentence_id, k)].fp32_units for k in range(7)))
    result = predict_features(FeatureBatch(tuple(rows), tuple(raw), tuple(prefixes),
        {'interface': 'supplied_native_prefix', 'feature_origin_verified': False}), models,
        budget_check=budget_check)
    native._unchanged(prefix_inputs)
    return result


def predict_components(rows, *, models, provider, budget_check=None):
    """Public raw-text adapter; the feature provider is an explicit dependency."""
    require(type(rows) in (tuple, list), 'ordered annotation-free RAW rows required')
    converted = []
    for row in rows:
        if type(row) is RawText:
            converted.append(row)
        else:
            require(type(row) is dict and set(row) == {'sentence_id', 'text', 'language'},
                    'only literal sentence_id/text/language fields are accepted')
            converted.append(RawText(**row))
    return predict_raw(tuple(converted), models=models, provider=provider, budget_check=budget_check)


class LegacyDomainError(ValueError):
    def __init__(self, report, result=None):
        super().__init__('legacy E2/CAN96/text4096 domain exceeded; full component result retained')
        self.report, self.component_result = report, result


def legacy_domain_report(result):
    require(type(result) is ComponentResult, 'complete current component result required')
    result.validate()
    events, mentions, texts = [], [], []
    for source, row in zip(result.features.rows, result.factors, strict=True):
        if len(source.text.encode('utf8')) > 4096:
            texts.append({'sentence_id': source.sentence_id, 'UTF8_bytes': len(source.text.encode('utf8'))})
        if len(row['events']) > 2:
            events.append({'sentence_id': source.sentence_id, 'predicted_events': len(row['events'])})
        for event in row['events']:
            for role in event['roles']:
                if len(role['surface'].encode('utf8')) > 96:
                    mentions.append({'sentence_id': source.sentence_id, 'event_index': event['event_index'],
                        'role_index': role['role_index'], 'UTF8_bytes': len(role['surface'].encode('utf8'))})
    return {'frozen_consumer_event_capacity': 2, 'CAN_input_byte_capacity': 96,
        'text_byte_capacity': 4096, 'consumer_capacity_failures': events,
        'CAN_domain_failures': mentions, 'text_domain_failures': texts,
        'compatible_with_legacy_boundaries': not (events or mentions or texts),
        'legacy_dense_basis_parity_claimed': False, 'qualification_claimed': False}


def require_legacy_compatible(result):
    report = legacy_domain_report(result)
    if not report['compatible_with_legacy_boundaries']:
        raise LegacyDomainError(report, result)
    return report
