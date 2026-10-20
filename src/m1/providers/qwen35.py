"""Exact frozen Qwen3.5-2B-Base C11 features for M1 Experimental 1.

No download, remote code, generation, chat template or training is performed.
The donor snapshot is separately obtained and authenticated before loading.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import unicodedata

import torch

from ..canonical_units import segment_text
from ..prefix_type_conditioning_layout_v1 import RawRow, Layout, align
from ..semantic_prefix_conditional_identity_contract_v1 import FORWARD_RECIPE, FORWARD_BASIS_SHA256
from ..runtime import RawText, FeatureBatch

REPOSITORY = 'Qwen/Qwen3.5-2B-Base'
REVISION = 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c'
EOS = 248044
LAYER = 11
SNAPSHOT_FILES = {
    'config.json': 'ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4',
    'merges.txt': 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d',
    'model.safetensors-00001-of-00001.safetensors': '928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d',
    'model.safetensors.index.json': '74d2ddfe79f10f35b27b498632f02b97b60dd9ec39b35d7c5a890c399284e319',
    'tokenizer.json': 'fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927',
    'tokenizer_config.json': 'e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de',
    'vocab.json': 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003',
}
_DTYPE_LOCK = threading.RLock()
_FP32_PATTERNS = (r'\.A_log$', r'\.linear_attn\.norm\.weight$')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def authenticate_snapshot(directory):
    root = Path(directory).resolve(strict=True)
    require(root.is_dir(), 'donor snapshot must be a local directory')
    for name, expected in SNAPSHOT_FILES.items():
        leaf = root / name
        require(leaf.is_file() and not leaf.is_symlink(), f'missing or linked donor file: {name}')
        digest = hashlib.sha256()
        with leaf.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        require(digest.hexdigest() == expected, f'pinned donor file SHA-256 differs: {name}')
    return root


@contextmanager
def _official_fp32_load_policy(model_class):
    # Transformers' dtype plan otherwise casts the official DeltaNet FP32 state.
    sentinel = object()
    with _DTYPE_LOCK:
        previous = model_class.__dict__.get('_keep_in_fp32_modules_strict', sentinel)
        model_class._keep_in_fp32_modules_strict = list(_FP32_PATTERNS)
        try:
            yield
        finally:
            if previous is sentinel:
                delattr(model_class, '_keep_in_fp32_modules_strict')
            else:
                model_class._keep_in_fp32_modules_strict = previous


def _validate_model(model, loading_info):
    require(not loading_info.get('error_msgs') and not loading_info.get('mismatched_keys'),
            'donor reported load errors or mismatched tensors')
    require(set(loading_info.get('missing_keys', ())) <= {'lm_head.weight'},
            'donor has missing text parameters')
    require(all(key.startswith(('model.visual.', 'mtp.'))
                for key in loading_info.get('unexpected_keys', ())),
            'donor reported unexpected text parameters')
    parameters = tuple(model.parameters())
    require(sum(p.numel() for p in parameters) == 1881825088
            and sum(p.numel() for p in parameters if p.dtype == torch.float32) == 2592
            and all(p.dtype in (torch.bfloat16, torch.float32) for p in parameters),
            'exact text-only BF16 / 2592 FP32 parameter inventory required')
    require(all(p.dtype == (torch.float32 if name.endswith(('.A_log', '.linear_attn.norm.weight'))
                           else torch.bfloat16) for name, p in model.named_parameters()),
            'official per-parameter BF16/FP32 policy differs')
    require(not any(any(term in name for term in ('visual', 'vision', 'mtp.', 'draft_model'))
                    for name, _ in model.named_parameters()), 'non-text parameters became resident')
    config = model.config
    require((config.hidden_size, config.num_hidden_layers, config.vocab_size) == (2048, 24, 248320)
            and config._attn_implementation == 'sdpa' and not getattr(model, '_use_kernels', False),
            'exact generic SDPA text donor topology required')
    # Optional fast packages can hide behind a generic-named wrapper. Reject them.
    for layer in model.model.layers:
        attention = getattr(layer, 'linear_attn', None)
        if attention is None:
            require(layer.block_type == 'full_attention', 'unrecognized donor mixer')
            forward = inspect.unwrap(type(layer.self_attn).forward)
            registry = forward.__globals__['ALL_ATTENTION_FUNCTIONS']
            resolved = registry.get_interface('sdpa', None)
            require(resolved.__module__ == 'transformers.integrations.sdpa_attention',
                    'generic SDPA dispatch required')
            continue
        _validate_generic_functions(attention)
    model.eval()
    for parameter in parameters:
        parameter.requires_grad_(False)


def _resolve_callable(module, name):
    pending, visited = [type(module).forward], set()
    while pending:
        candidate = pending.pop()
        if not callable(candidate) or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        unwrapped = inspect.unwrap(candidate)
        if id(unwrapped) not in visited:
            pending.append(unwrapped)
        resolved = getattr(candidate, '__globals__', {}).get(name)
        if callable(resolved):
            return resolved
        try:
            closure = inspect.getclosurevars(candidate)
        except (TypeError, ValueError):
            continue
        pending.extend(value for value in closure.nonlocals.values() if callable(value))
    raise ValueError(f'generic DeltaNet callable absent: {name}')


def _validate_generic_functions(attention):
    hidden = getattr(attention, '_kernel_funcs', {})
    names = {'causal_conv1d_fn': 'causal_conv1d_fn', 'causal_conv1d_update': 'causal_conv1d_update',
             'chunk_gated_delta_rule': 'torch_chunk_gated_delta_rule',
             'recurrent_gated_delta_rule': 'torch_recurrent_gated_delta_rule'}
    for registered, name in names.items():
        function = _generic_function(_resolve_callable(attention, name))
        require(function.__module__ == 'transformers.models.qwen3_5.modeling_qwen3_5'
                and function.__name__ == name, f'non-generic fallback implementation: {name}')
        dispatched = hidden.get(registered)
        if dispatched is not None:
            # kernels registers Func wrappers even with use_kernels=False.
            # Their closure must point to the exact generic fallback above.
            target = _generic_function(dispatched)
            require(target is function, f'non-generic DeltaNet dispatch: {registered}')


def _generic_function(value):
    visited = set()
    while id(value) not in visited:
        visited.add(id(value))
        forward = getattr(value, 'forward', value)
        closure = inspect.getclosurevars(forward)
        if 'func' in closure.nonlocals:
            value = closure.nonlocals['func']
            continue
        implementation = closure.nonlocals.get('implementation')
        require(implementation is None or implementation is inspect.unwrap(forward),
                'optional implementation behind generic DeltaNet callable')
        return forward
    raise ValueError('cyclic DeltaNet function wrapper')


def _configure_math():
    require(os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') != '1', 'TF32 override forbidden')
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _tokenize(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, padding=False, truncation=False,
                        return_attention_mask=False, return_offsets_mapping=True)
    ids = tuple(encoded['input_ids'])
    offsets = tuple(tuple(pair) for pair in encoded['offset_mapping'])
    require(ids and len(ids) == len(offsets) and EOS not in ids and len(ids) + 1 <= 512,
            'nonempty original text plus one EOS must fit 512 tokens; no truncation')
    require(all(type(i) is int and i >= 0 for i in ids)
            and all(type(s) is type(e) is int and 0 <= s <= e <= len(text) for s, e in offsets),
            'tokenizer IDs or Unicode offsets differ')
    return ids, offsets


def _plan(rows, tokenizer):
    require(unicodedata.unidata_version == '15.0.0', 'Unicode 15.0.0 required (tested Python 3.12)')
    require(type(rows) is tuple and rows and all(type(row) is RawText for row in rows)
            and len({row.sentence_id for row in rows}) == len(rows), 'unique annotation-free RawText tuple required')
    raw_items, prefixed_items = [], []
    for row in rows:
        units = tuple((u.char_start, u.char_end) for u in row.validate())
        source = RawRow(row.sentence_id, row.text, row.language, units)
        ids, offsets = _tokenize(tokenizer, row.text)
        raw_indices = tuple(tuple(i for i, (a, b) in enumerate(offsets)
                                  if b > a and b > s and a < e) for s, e in units)
        require(all(raw_indices), 'every canonical unit must overlap a source token')
        raw_items.append((ids, raw_indices))
        for candidate, prefix in enumerate(FORWARD_RECIPE['prefixes']):
            layout = Layout(source, 'meaning_prefix', candidate, prefix + row.text,
                            len(prefix), FORWARD_BASIS_SHA256)
            typed_ids, typed_offsets = _tokenize(tokenizer, layout.encoded_text)
            aligned = align(layout, typed_ids, typed_offsets, ids, offsets)
            prefixed_items.append((typed_ids, aligned.unit_token_indices))
    return tuple(raw_items), tuple(prefixed_items)


class _ReachedLayer(Exception):
    def __init__(self, token, hidden):
        self.token, self.hidden = token, hidden


def _capture_batch(model, items, device):
    lengths = tuple(len(ids) + 1 for ids, _ in items)
    ids = torch.full((len(items), max(lengths)), EOS, dtype=torch.int64, device=device)
    mask = torch.zeros_like(ids)
    for row, ((tokens, _), length) in enumerate(zip(items, lengths, strict=True)):
        ids[row, :length] = torch.tensor((*tokens, EOS), dtype=torch.int64, device=device)
        mask[row, :length] = 1
    marker = object()
    def hook(_module, _inputs, output):
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        raise _ReachedLayer(marker, hidden)
    handle = model.model.layers[LAYER].register_forward_hook(hook)
    hidden = None
    try:
        try:
            with torch.inference_mode():
                model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
        except _ReachedLayer as signal:
            if signal.token is not marker:
                raise
            hidden = signal.hidden
    finally:
        handle.remove()
    require(hidden is not None and tuple(hidden.shape) == (*ids.shape, 2048),
            'exact post-block residual C11 capture required')
    with torch.inference_mode(False):
        states = hidden.detach().to(device='cpu', dtype=torch.float32).clone().contiguous()
    require(bool(states.isfinite().all()), 'nonfinite donor states')
    return tuple(torch.stack([states[row, list(indices)].mean(dim=0, dtype=torch.float32)
                              for indices in unit_indices]).contiguous()
                 for row, (_, unit_indices) in enumerate(items))


class Qwen35FeatureProvider:
    """Pinned, local, text-only donor; context manager releases owned resources.

    Loading configures deterministic PyTorch math. Calls are serialized because
    a scoped hook exits at block11. All donor tensors remain frozen.
    """
    def __init__(self, snapshot, *, device='cpu', max_units=128):
        require(type(max_units) is int and max_units > 0, 'positive explicit unit budget required')
        self._lock = threading.RLock()
        self.max_units = max_units
        self.device = torch.device(device)
        self.model = self.tokenizer = None
        root = authenticate_snapshot(snapshot)
        _configure_math()
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, AutoTokenizer
        payload = json.loads((root / 'config.json').read_text(encoding='utf-8'))['text_config']
        config = Qwen3_5TextConfig.from_dict(payload)
        config.use_cache = False
        config._attn_implementation = 'sdpa'
        with _official_fp32_load_policy(Qwen3_5ForCausalLM):
            model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
                str(root), config=config, dtype=torch.bfloat16, local_files_only=True,
                trust_remote_code=False, low_cpu_mem_usage=True, device_map={'': str(self.device)},
                use_kernels=False, kernel_config=None,
                key_mapping={r'^model\.language_model\.': 'model.'}, output_loading_info=True)
        _validate_model(model, loading_info)
        tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True,
            trust_remote_code=False, use_fast=True, padding_side='right')
        require(tokenizer.is_fast and tokenizer.padding_side == 'right' and tokenizer.eos_token_id == EOS,
                'exact fast tokenizer / right padding / EOS required')
        self.model, self.tokenizer = model, tokenizer

    def capture(self, rows):
        with self._lock:
            require(self.model is not None and self.tokenizer is not None, 'provider is closed')
            require(type(rows) is tuple and rows and all(type(row) is RawText for row in rows),
                    'annotation-free RawText tuple required')
            for row in rows:
                require(len(row.validate()) <= self.max_units, 'input exceeds explicit canonical-unit budget')
            raw_plan, prefix_plan = _plan(rows, self.tokenizer)
            before = tuple((p.data_ptr(), p._version) for p in self.model.parameters())
            require(not self.model.training and all(not p.requires_grad for p in self.model.parameters()),
                    'frozen eval donor required')
            raw, prefixes = [], []
            for items, sink in ((raw_plan, raw), (prefix_plan, prefixes)):
                for first in range(0, len(items), 8):
                    sink.extend(_capture_batch(self.model, items[first:first + 8], self.device))
            require(before == tuple((p.data_ptr(), p._version) for p in self.model.parameters()),
                    'donor changed during feature capture')
            return FeatureBatch(rows, tuple(raw), tuple(tuple(prefixes[n*7:(n+1)*7])
                for n in range(len(rows))), {'provider': 'QWEN35_C11_v1', 'repository': REPOSITORY,
                'revision': REVISION, 'weight_sha256': SNAPSHOT_FILES['model.safetensors-00001-of-00001.safetensors'],
                'tokenizer_sha256': SNAPSHOT_FILES['tokenizer.json'],
                'prefix_basis_sha256': FORWARD_BASIS_SHA256, 'layer_zero_based': LAYER,
                'normalization': 'NONE', 'Gold_inputs': 0, 'donor_updates': 0})

    def close(self):
        with self._lock:
            self.model = self.tokenizer = None
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
