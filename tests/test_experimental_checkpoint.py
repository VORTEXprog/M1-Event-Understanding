"""Additive CAN domain/routing and authenticated release loading boundaries."""
import hashlib
import json

import pytest
import torch

from m1 import checkpoint
from m1.experimental import load_experimental_checkpoint
from m1 import context_canonicalizer_input128 as can128
from m1 import context_canonicalizer_routed as routed


def state32():
    # Explicit invented test tensors, never a pretrained model fallback.
    return {k:torch.zeros(shape,dtype=torch.float32) for k,shape in routed.STATE_SHAPES.items()}


def test_can128_has_literal_utf8_caps_and_unmodified_context_slices():
    text='left '+'я'*64+' right'
    batch=can128.encode_mentions((can128.ContextMention(text,5,69),))
    assert batch.input_lengths.tolist()==[128]
    assert bytes(batch.input_byte_ids[0].tolist()).decode('utf8')=='я'*64
    assert bytes(batch.left_byte_ids[0,:int(batch.left_lengths[0])].tolist())==b'left '
    with pytest.raises(ValueError,match='cap128'):
        can128.encode_mentions((can128.ContextMention('a'*129,0,129),))
    with pytest.raises(ValueError,match='4096'):
        can128.encode_mentions((can128.ContextMention('a'*4097,0,1),))


def test_router_restores_all32_privately_and_keeps_native_decode_and_ties():
    state=state32();rng=torch.get_rng_state().clone()
    owner=routed.model_from_state(state)
    assert torch.equal(rng,torch.get_rng_state())
    actual=routed.complete_state(owner)
    assert len(actual)==32
    assert all(torch.equal(v,actual[k])and v.data_ptr()!=actual[k].data_ptr()for k,v in state.items())
    mentions=(can128.ContextMention('prefix alpha suffix',7,12),)
    decoded,routes,logits=owner.decode_with_routes(mentions)
    assert routes==(routed.FULL,)and torch.equal(logits,torch.zeros(1,2))
    assert decoded==owner.canonicalizer.decode_mentions(mentions)
    state['route_output.bias'][1]=2.
    other=routed.model_from_state(state)
    decoded,routes,_=other.decode_with_routes(mentions)
    assert routes==(routed.ONE_LEFT_BLANK,)
    assert decoded==other.canonicalizer.decode_mentions((can128.ContextMention(' alpha',1,6),))
    assert all(not p.requires_grad and p.grad is None for p in other.parameters())
    assert torch.equal(rng,torch.get_rng_state())


def test_top_hash_is_required_before_any_bundle_or_model_parse(tmp_path):
    content=b'{"untrusted":true}\n';(tmp_path/'CHECKPOINT.json').write_bytes(content)
    with pytest.raises(ValueError,match='manifest SHA256'):
        load_experimental_checkpoint(tmp_path,expected_manifest_sha256='0'*64)
    with pytest.raises(ValueError,match='closed compatible'):
        load_experimental_checkpoint(tmp_path,expected_manifest_sha256=hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError,match='commitment'):
        load_experimental_checkpoint(tmp_path,expected_manifest_sha256='')


def test_router_rejects_partial_state_and_execution_hooks():
    state=state32();incomplete=dict(state);incomplete.pop('route_output.bias')
    with pytest.raises(ValueError,match='complete32'):
        routed.model_from_state(incomplete)
    owner=routed.model_from_state(state)
    handle=owner.route_output.register_forward_hook(lambda m,i,o:o)
    try:
        with pytest.raises(ValueError,match='hooks'):
            routed.validate_model(owner,learning=False)
    finally:
        handle.remove()
