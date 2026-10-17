"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from dataclasses import dataclass

import hashlib

import json

VERSION = 'M1_AM190_PREFIX_TYPE_CONDITIONING_LAYOUT_v1'

TYPES = ('PURCHASE', 'GIVE', 'OWN', 'OPEN', 'CLOSE', 'REPORT', 'SHOW')

ARMS = ('unprimed', 'meaning_prefix', 'neutral_prefix')

DEFINITIONS = (
    'Receiving ownership of an item in exchange for payment.',
    'Permanently transferring ownership of an item to another person without payment.',
    'Having possession or ownership of an item as a state.',
    'Changing an object from a closed state to an open state.',
    'Changing an object from an open state to a closed state.',
    'Communicating information or a proposition to another person.',
    'Intentionally making an item visible to another person without transferring ownership.',
)

HEADER, DELIMITER = 'Candidate meaning: ', '\n\nText:\n'

MAX_TOKENS = 512

DONOR_REVISION = 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c'

DONOR_WEIGHT_SHA256 = '928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d'

TOKENIZER_SHA256 = 'fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927'

def require(value, message):
    if not value:
        raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

ORIGINAL_BASIS_SHA256 = digest({'donor_revision': DONOR_REVISION,
    'weight_sha256': DONOR_WEIGHT_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256,
    'selected_layer_zero_based': 11, 'source_text': 'ORIGINAL_UNPREFIXED_RAW_TEXT',
    'special_tokens': False, 'one_appended_eos': 248044,
    'pool': 'RAW_FP32_OVERLAP_MEAN', 'normalization': 'NONE',
    'production_feature_bank_issuer': False})

@dataclass(frozen=True)
class RawRow:
    sentence_id: str
    text: str
    language: str
    units: tuple[tuple[int, int], ...]

    def __post_init__(self):
        require(type(self.sentence_id) is str and self.sentence_id and type(self.text) is str
            and self.text and self.language in ('en', 'ru') and self.units, 'raw row required')
        previous = 0
        for start, end in self.units:
            require(type(start) is type(end) is int and previous <= start < end <= len(self.text),
                'ordered original codepoint units required')
            previous = end

@dataclass(frozen=True)
class Recipe:
    neutral_prefixes: tuple[str, ...]
    prefix_token_counts: tuple[int, ...]
    tokenizer_sha256: str

    def __post_init__(self):
        require(len(self.neutral_prefixes) == len(self.prefix_token_counts) == 7
            and len(self.tokenizer_sha256) == 64
            and not set(self.tokenizer_sha256) - set('0123456789abcdef'), 'exact recipe7 required')
        for p, n in zip(self.neutral_prefixes, self.prefix_token_counts, strict=True):
            require(type(p) is str and p.startswith(HEADER) and p.endswith(DELIMITER)
                and set(p[len(HEADER):-len(DELIMITER)].split()) == {'x'}
                and type(n) is int and 0 < n < MAX_TOKENS, 'neutral x-only prefix required')

    @property
    def basis_sha256(self):
        return digest({'version': VERSION, 'candidate_order': TYPES, 'definitions': DEFINITIONS,
            'donor_revision': DONOR_REVISION, 'weight_sha256': DONOR_WEIGHT_SHA256,
            'neutral_prefixes': self.neutral_prefixes, 'prefix_token_counts': self.prefix_token_counts,
            'tokenizer_sha256': self.tokenizer_sha256, 'selected_layer_zero_based': 11,
            'special_tokens': False, 'one_appended_eos': 248044, 'max_valid_tokens': MAX_TOKENS,
            'source_units': 'ORIGINAL_CODEPOINT_SPANS_ONLY', 'pool': 'RAW_FP32_OVERLAP_MEAN',
            'normalization': 'NONE', 'production_feature_bank_issuer': False})

def meaningful_prefix(candidate):
    require(type(candidate) is int and 0 <= candidate < 7, 'all7 native candidate index required')
    return HEADER + DEFINITIONS[candidate] + DELIMITER

def build_recipe(tokenize, *, tokenizer_sha256):
    """Tokenizer-only fixed x-repeat search, never model/score selection."""
    neutral, counts = [], []
    for k in range(7):
        target = len(tokenize(meaningful_prefix(k))[0])
        matches = []
        for n in range(1, MAX_TOKENS):
            value = HEADER + ' '.join(['x'] * n) + DELIMITER
            if len(tokenize(value)[0]) == target:
                matches.append(value)
                break
        require(matches, 'neutral prefix cannot match tokenizer length; no fallback')
        neutral.append(matches[0]); counts.append(target)
    return Recipe(tuple(neutral), tuple(counts), tokenizer_sha256)

@dataclass(frozen=True)
class Layout:
    row: RawRow
    arm: str
    candidate: int
    encoded_text: str
    source_start: int
    basis_sha256: str

    def __post_init__(self):
        require(self.arm in ARMS and type(self.candidate) is int and 0 <= self.candidate < 7
            and type(self.source_start) is int and self.source_start >= 0
            and self.encoded_text[self.source_start:] == self.row.text,
            'exact original raw-text suffix and dense candidate required')

    @property
    def item_id(self):
        return digest([self.row.sentence_id, self.arm, self.candidate, self.basis_sha256])

def render(row, recipe, arm, candidate):
    require(type(row) is RawRow and type(recipe) is Recipe and arm in ARMS, 'typed separate-basis ingress required')
    meaningful_prefix(candidate)  # validate candidate even for unprimed
    prefix = '' if arm == 'unprimed' else meaningful_prefix(candidate) if arm == 'meaning_prefix' else recipe.neutral_prefixes[candidate]
    basis = ORIGINAL_BASIS_SHA256 if arm == 'unprimed' else recipe.basis_sha256
    return Layout(row, arm, candidate, prefix + row.text, len(prefix), basis)

@dataclass(frozen=True)
class Alignment:
    token_ids: tuple[int, ...]
    token_offsets: tuple[tuple[int, int], ...]
    source_token_indices: tuple[int, ...]
    original_offsets: tuple[tuple[int, int], ...]
    unit_token_indices: tuple[tuple[int, ...], ...]

def align(layout, token_ids, token_offsets, original_ids, original_offsets):
    """Reject boundary merges, missing units, token-ID drift and CHAR drift."""
    require(type(layout) is Layout and len(token_ids) == len(token_offsets)
        and len(original_ids) == len(original_offsets), 'exact tokenizer coordinates required')
    ids, offsets = tuple(token_ids), tuple(tuple(p) for p in token_offsets)
    require(ids and all(type(x) is int and 0 <= x != 248044 for x in ids)
        and all(type(x) is int and 0 <= x != 248044 for x in original_ids), 'ordinary non-EOS token IDs required')
    require(all(type(s) is type(e) is int and 0 <= s < e <= len(layout.row.text)
        for s, e in original_offsets), 'nonempty original codepoint token offsets required')
    selected, mapped = [], []
    for i, (start, end) in enumerate(offsets):
        require(type(start) is type(end) is int and 0 <= start <= end <= len(layout.encoded_text),
            'valid complete encoded codepoint offsets required')
        require(not start < layout.source_start < end, 'prefix/source crossing token forbidden')
        if end > start and start >= layout.source_start:
            selected.append(i); mapped.append((start-layout.source_start, end-layout.source_start))
    require(tuple(ids[i] for i in selected) == tuple(original_ids)
        and tuple(mapped) == tuple(tuple(p) for p in original_offsets),
        'original suffix token IDs and CHAR offsets must be exact; no trimming repair')
    unit_tokens = []
    for left, right in layout.row.units:
        indices = tuple(i for i, (start, end) in zip(selected, mapped, strict=True) if end > left and start < right)
        require(indices, 'every original canonical unit needs a captured source token')
        unit_tokens.append(indices)
    require(len(ids) + 1 <= MAX_TOKENS, 'prefix plus source plus EOS exceeds native512; no truncation')
    return Alignment(ids, offsets, tuple(selected), tuple(mapped), tuple(unit_tokens))

def plan_rows(rows, recipe, tokenize):
    """Commit complete dense layout/token plan before any F32/model ingress."""
    require(rows and len({r.sentence_id for r in rows}) == len(rows), 'unique selected raw rows required')
    plan = []
    for row in rows:
        original_ids, original_offsets = tokenize(row.text)
        for arm in ARMS:
            for k in range(7):
                layout = render(row, recipe, arm, k)
                ids, offsets = tokenize(layout.encoded_text)
                alignment = align(layout, ids, offsets, original_ids, original_offsets)
                if arm != 'unprimed':
                    require(len(ids)-len(original_ids) == recipe.prefix_token_counts[k], 'full-prefix token count differs')
                plan.append((layout, alignment))
    return tuple(plan)

def plan_commitment(plan):
    return digest([{'item_id': l.item_id, 'sentence_id': l.row.sentence_id, 'arm': l.arm,
        'candidate': l.candidate, 'basis_sha256': l.basis_sha256, 'source_start': l.source_start,
        'encoded_text_sha256': hashlib.sha256(l.encoded_text.encode()).hexdigest(),
        'token_ids': a.token_ids, 'token_offsets': a.token_offsets,
        'unit_token_indices': a.unit_token_indices} for l, a in plan])

def physical_batches(plan, batch_size=8):
    """Arm-wise paired batch geometry; unprimed is encoded once and reused7."""
    require(batch_size == 8 and plan, 'fixed batch8 and complete plan required')
    rows = tuple(dict.fromkeys(l.row.sentence_id for l,a in plan))
    expected = {(r, arm, k) for r in rows for arm in ARMS for k in range(7)}
    observed = [(l.row.sentence_id,l.arm,l.candidate) for l,a in plan]
    require(len(observed) == len(expected) and set(observed) == expected, 'complete dense all7/three-arm plan required')
    grouped = {r: [p for p in plan if p[0].row.sentence_id == r] for r in rows}
    for group in grouped.values():
        require(all(l.row == group[0][0].row for l,a in group), 'one immutable original raw row across all candidates required')
        unprimed = [(l,a) for l,a in group if l.arm == 'unprimed']
        require(all(l.basis_sha256 == ORIGINAL_BASIS_SHA256 and l.encoded_text == l.row.text
            and a == unprimed[0][1] for l,a in unprimed), 'unprimed reuse token/geometry/basis differs')
        for k in range(7):
            pairs = [p for p in group if p[0].candidate == k and p[0].arm != 'unprimed']
            require(len(pairs[0][1].token_ids) == len(pairs[1][1].token_ids), 'semantic/neutral paired padding geometry differs')
    physical = {arm: [p for p in plan if p[0].arm == arm
        and (arm != 'unprimed' or p[0].candidate == 0)] for arm in ARMS}
    for arm in ARMS:
        for start in range(0,len(physical[arm]),batch_size):
            yield tuple(physical[arm][start:start+batch_size])
