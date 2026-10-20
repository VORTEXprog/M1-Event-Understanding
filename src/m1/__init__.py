"""Current-turn structured inference research toolkit; trained weights are external."""
__version__ = '0.1.0a1'
from .runtime import (
    RawText, FeatureBatch, FeatureProvider, SuppliedComponents, ComponentResult,
    predict_features, predict_raw, predict_components, compose_components,
    replay_saved_binding, LegacyDomainError, legacy_domain_report, require_legacy_compatible,
)
from .factories import (
    components_from_states, experimental_components, export_component_states,
    original66_from_state, head96_from_state, binding_from_state,
    canonicalizer_from_state, slots_from_state,
)
from .canonical_units import CanonicalUnit, segment_text
from .semantic_prefix_conditional_identity_contract_v1 import TYPES
from .heads import ROLE_ORDER, POLARITY_ORDER, RELATION_ORDER

__all__ = [
    'RawText', 'FeatureBatch', 'FeatureProvider', 'SuppliedComponents', 'ComponentResult',
    'predict_features', 'predict_raw', 'predict_components', 'compose_components',
    'replay_saved_binding', 'LegacyDomainError', 'legacy_domain_report', 'require_legacy_compatible',
    'components_from_states', 'experimental_components', 'export_component_states',
    'original66_from_state', 'head96_from_state', 'binding_from_state',
    'canonicalizer_from_state', 'slots_from_state', 'CanonicalUnit', 'segment_text',
    'TYPES', 'ROLE_ORDER', 'POLARITY_ORDER', 'RELATION_ORDER',
]

from .experimental import (load_experimental_checkpoint, ExperimentalCheckpoint,
    canonicalizer128_from_state, routed_canonicalizer_from_state, canonicalize_components)
__all__ += ['load_experimental_checkpoint', 'ExperimentalCheckpoint',
    'canonicalizer128_from_state', 'routed_canonicalizer_from_state', 'canonicalize_components']
