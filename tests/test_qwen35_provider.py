"""Invented CPU fixtures for provider plumbing; no weights or downloads."""
import hashlib
import threading
from types import SimpleNamespace

import pytest
import torch

from m1.runtime import RawText
from m1.providers import qwen35 as q


class CharacterTokenizer:
    def __call__(self, text, **_kwargs):
        return {'input_ids': [ord(c) for c in text],
                'offset_mapping': [(n, n + 1) for n in range(len(text))]}


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        return hidden + 1


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer() for _ in range(24)])

    def forward(self, *, input_ids, attention_mask, **_kwargs):
        hidden = input_ids.float()[..., None].expand(-1, -1, 2048).clone()
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def invented_provider():
    provider = q.Qwen35FeatureProvider.__new__(q.Qwen35FeatureProvider)
    provider._lock = threading.RLock()
    provider.device = torch.device('cpu')
    provider.max_units = 128
    provider.tokenizer = CharacterTokenizer()
    provider.model = torch.nn.Module()
    provider.model.model = Backbone()
    provider.model.eval()
    return provider


def test_complete_c11_raw_and_seven_bank_capture():
    torch.set_num_threads(2)
    rows = (RawText('invented/en', 'A opened it.', 'en'),
            RawText('invented/ru', 'Б закрыл ящик.', 'ru'))
    provider = invented_provider()
    before = torch.random.get_rng_state().clone()
    result = provider.capture(rows)
    result.validate()
    assert torch.equal(before, torch.random.get_rng_state())
    assert provider.model.model.layers[11].calls == 3  # raw2, then prefix8+6
    assert provider.model.model.layers[12].calls == 0
    assert len(result.unprimed) == len(result.prefixes) == 2
    for row, raw, prefixed in zip(rows, result.unprimed, result.prefixes):
        expected = torch.tensor([sum(map(ord, u.text))/len(u.text) + 12
                                 for u in row.validate()], dtype=torch.float32)
        assert torch.allclose(raw[:, 0], expected, atol=1e-5)
        assert len(prefixed) == 7
        assert all(torch.equal(raw, value) for value in prefixed)
        assert all(not value.is_inference() for value in (raw, *prefixed))
        assert len({v.untyped_storage().data_ptr() for v in (raw, *prefixed)}) == 8
    provider.close()
    with pytest.raises(ValueError, match='closed'):
        provider.capture(rows)


def test_exact_prefixes_full_units_alignment_and_explicit_overflow():
    row = RawText('invented/a', 'Иван — здесь.', 'ru')
    raw, prefixes = q._plan((row,), CharacterTokenizer())
    assert len(raw) == 1 and len(prefixes) == 7
    assert len(prefixes[0][1]) == len(row.validate())  # punctuation and suffix retained
    expected = q.FORWARD_RECIPE['prefixes'][0] + row.text
    assert prefixes[0][0] == tuple(map(ord, expected))
    with pytest.raises(ValueError, match='512'):
        q._plan((RawText('invented/long', 'а' * 500, 'ru'),), CharacterTokenizer())
    with pytest.raises(ValueError, match='unique'):
        q._plan((row, row), CharacterTokenizer())
    class DriftTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if text.startswith('Candidate meaning: '):
                result['input_ids'][-1] += 1
            return result
    with pytest.raises(ValueError, match='suffix token IDs'):
        q._plan((row,), DriftTokenizer())


def test_snapshot_rejects_wrong_bytes_and_missing_files(tmp_path, monkeypatch):
    leaf = tmp_path / 'tokenizer.json'
    leaf.write_bytes(b'invented')
    monkeypatch.setattr(q, 'SNAPSHOT_FILES', {'tokenizer.json': hashlib.sha256(b'invented').hexdigest()})
    assert q.authenticate_snapshot(tmp_path) == tmp_path
    leaf.write_bytes(b'different')
    with pytest.raises(ValueError, match='SHA-256'):
        q.authenticate_snapshot(tmp_path)
    leaf.unlink()
    with pytest.raises(ValueError, match='missing'):
        q.authenticate_snapshot(tmp_path)


def test_unit_budget_and_unicode_policy(monkeypatch):
    provider = invented_provider()
    provider.max_units = 1
    with pytest.raises(ValueError, match='unit budget'):
        provider.capture((RawText('invented/u', 'two units', 'en'),))
    monkeypatch.setattr(q.unicodedata, 'unidata_version', 'invented')
    with pytest.raises(ValueError, match='Unicode 15'):
        q._plan((RawText('invented/u', 'A', 'en'),), CharacterTokenizer())


def test_generic_registered_wrappers_are_not_accelerated_kernels(monkeypatch):
    def generic(*args):
        return args
    class Wrapper:
        def __init__(self, func):
            def forward(*args):
                return func(*args)
            self.forward = forward
    names = ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule', 'recurrent_gated_delta_rule')
    generics = {}
    for name in names:
        def function(*args):
            return args
        function.__name__ = name if name.startswith('causal_') else 'torch_' + name
        function.__module__ = 'transformers.models.qwen3_5.modeling_qwen3_5'
        generics[function.__name__] = function
    monkeypatch.setattr(q, '_resolve_callable', lambda _module, name: Wrapper(generics[name]))
    attention = SimpleNamespace(_kernel_funcs={name: Wrapper(generics[name if name.startswith('causal_')
                                                              else 'torch_' + name]) for name in names})
    q._validate_generic_functions(attention)
    attention._kernel_funcs['chunk_gated_delta_rule'] = Wrapper(lambda: None)
    with pytest.raises(ValueError, match='non-generic'):
        q._validate_generic_functions(attention)
