"""Authenticated Experimental1 bundles and optional learned canonicalization.

Loading establishes tensor integrity and architecture compatibility. It does
not certify model quality, independent training origins or release approval.
"""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from . import checkpoint, factories
from . import context_canonicalizer_input128 as can128
from . import context_canonicalizer_routed as routed

FORMAT = 'M1_EXPERIMENTAL_CHECKPOINT_v1'
# Membership is descriptive only: no alias lookup, fallback, filtering or rewrite.
FINITE27_IDS = frozenset('entity/' + name for name in (
    'Cedric','Nolan','Wallace','Harvey','Quentin','Desmond','Bennett','Conrad',
    'Duncan','Gareth','Howard','Russell','Clifford','Donovan','Gerard','Neville',
    'Roderick','Wilfred')) | frozenset('object/' + name for name in (
    'trunk_sunduk','cabinet_shkaf','knapsack_ranets','hamper_korob','cupboard_bufet',
    'toolbox_yashchik','locker_shkafchik','bin_konteyner','barrel_bochonok'))


def canonicalizer128_from_state(state):
    return can128.model_from_state(state)


def routed_canonicalizer_from_state(state):
    return routed.model_from_state(state)


@dataclass(frozen=True)
class ExperimentalCheckpoint:
    models: object
    canonicalizer: object
    slots: object
    manifest: dict
    manifest_sha256: str


def load_experimental_checkpoint(directory, *, expected_manifest_sha256):
    """Load only authenticated data, then explicitly restore frozen owners.

    The expected top-level SHA256 must be obtained independently of the bundle.
    A digest stored inside the downloaded bundle is not its own trust anchor.
    """
    checkpoint._expected(expected_manifest_sha256)
    directory = checkpoint._directory(directory)
    payload = checkpoint._read(directory / 'CHECKPOINT.json', checkpoint.MAX_MANIFEST_BYTES)
    if hashlib.sha256(payload).hexdigest() != expected_manifest_sha256:
        raise ValueError('checkpoint manifest SHA256 mismatch')
    document = json.loads(payload)
    if (type(document) is not dict or set(document) != {
            'format','name','experimental','qualified','architecture','bundles','metadata'}
            or document['format'] != FORMAT or document['experimental'] is not True
            or document['qualified'] is not False or document['architecture'] != {
                'components': 'type14_loc13_original66_heads96_binding83',
                'canonicalizer': 'routed_CAN32_input128', 'slots': 'slots4'}
            or type(document['bundles']) is not dict
            or set(document['bundles']) != {'components','canonicalizer','slots'}):
        raise ValueError('closed compatible experimental checkpoint manifest required')
    checkpoint._json_value(document['metadata'])
    loaded = {}
    expected_groups = {'components':set(factories.COMPONENT_NAMES),
                       'canonicalizer':{'canonicalizer'}, 'slots':{'slots'}}
    for name in ('components','canonicalizer','slots'):
        spec = document['bundles'][name]
        if type(spec) is not dict or set(spec) != {
                'archive_sha256','manifest_sha256','state_sha256'}:
            raise ValueError('closed independently committed bundle record required')
        item = checkpoint.load_state_bundle(directory/name,
            expected_archive_sha256=spec['archive_sha256'],
            expected_manifest_sha256=spec['manifest_sha256'])
        if set(item.states) != expected_groups[name]:
            raise ValueError('exact experimental state groups required')
        if (type(spec['state_sha256']) is not dict
                or set(spec['state_sha256']) != expected_groups[name]
                or any(factories.state_dict_sha256(state) != spec['state_sha256'][group]
                       for group,state in item.states.items())):
            raise ValueError('complete experimental tensor state digest mismatch')
        loaded[name] = item.states
    models = factories.components_from_states(loaded['components'],
        expected_state_sha256=document['bundles']['components']['state_sha256'])
    canonicalizer = routed_canonicalizer_from_state(loaded['canonicalizer']['canonicalizer'])
    slots = factories.slots_from_state(loaded['slots']['slots'])
    return ExperimentalCheckpoint(models,canonicalizer,slots,deepcopy(document),expected_manifest_sha256)


def canonicalize_components(result, canonicalizer, *, batch_rows=128):
    """Annotate actual predicted role spans with learned byte-decoder outputs.

    Inputs must be a real component result. No supplied Gold spans, entity IDs
    or labels enter this operation. Returned IDs remain the exact learned
    strings, including predictions outside the declared finite27 inventory.
    Role geometry beyond the CAN128/text4096 domain is retained and marked.
    This helper does not run slots or any application-specific consumer.
    """
    from .runtime import ComponentResult, require
    require(type(result) is ComponentResult, 'complete predicted component result required')
    result.validate()
    routed.validate_model(canonicalizer,learning=False)
    require(type(batch_rows) is int and 0 < batch_rows <= 128, 'CAN batch size1..128 required')
    before = routed.state_digest(routed.complete_state(canonicalizer))
    rows = deepcopy(result.factors)
    mentions, positions = [], []
    for source,row in zip(result.features.rows, rows, strict=True):
        require(source.sentence_id == row['sentence_id'], 'predicted row order differs')
        for event in row['events']:
            for role in event['roles']:
                start,end = role['char_start'],role['char_end']
                require(type(start) is type(end) is int and 0 <= start < end <= len(source.text)
                    and source.text[start:end] == role['surface'], 'literal predicted role span required')
                if len(source.text.encode('utf8')) > 4096 or len(role['surface'].encode('utf8')) > 128:
                    role.update(canonical_id=None,canonical_status='OUTSIDE_CAN_DOMAIN',
                                canonical_inventory27=None)
                else:
                    positions.append(role)
                    mentions.append(can128.ContextMention(source.text,start,end))
    for begin in range(0,len(mentions),batch_rows):
        decoded,routes,_ = canonicalizer.decode_with_routes(tuple(mentions[begin:begin+batch_rows]))
        for role,value,route in zip(positions[begin:begin+batch_rows],decoded,routes,strict=True):
            role.update(canonical_id=value.text,canonical_status='DECODED',
                        canonical_inventory27=value.text in FINITE27_IDS,
                        canonical_route='FULL' if route == routed.FULL else 'ONE_LEFT_BLANK')
    require(routed.state_digest(routed.complete_state(canonicalizer)) == before,
            'canonicalizer changed during inference')
    result.validate()
    return rows
