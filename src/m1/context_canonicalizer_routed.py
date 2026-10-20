"""Learned full-context/one-left-blank routing around frozen CAN28.

Only raw surface and context representations enter the router. Native decode
runs once after routing; canonical IDs and decoded outputs never select a route.
"""
import torch
from torch import nn
from . import context_canonicalizer_input128 as native
FULL,ONE_LEFT_BLANK=0,1
ROUTE_SHAPES={'route_hidden.weight':(16,768),'route_hidden.bias':(16,),
    'route_output.weight':(2,16),'route_output.bias':(2,)}
STATE_SHAPES={**native.STATE_SHAPES,**ROUTE_SHAPES}
PARAMETER_KEYS,PARAMETER_COUNT,TRAINABLE_PARAMETERS=32,870332,12338
require=native.require


def validate_state(state):
    require(type(state) is dict and set(state)==set(STATE_SHAPES), 'exact complete32 state required')
    native.validate_state({k:state[k] for k in native.STATE_SHAPES})
    require(all(type(v) is torch.Tensor and tuple(v.shape)==STATE_SHAPES[k]
        and v.dtype==torch.float32 and v.device.type=='cpu' and v.layout==torch.strided
        and v.is_contiguous() and not v.requires_grad and not v.is_inference()
        and bool(torch.isfinite(v).all()) for k,v in state.items()), 'ordinary finite private CPU FP32 shapes required')
    require(len({v.untyped_storage().data_ptr() for v in state.values()})==32, 'all32 private storage required')

def state_digest(state):
    validate_state(state)
    return native.native._state_digest(state)

class RoutedCAN32(nn.Module):
    def __init__(self, frozen28):
        super().__init__()
        self.canonicalizer = native.model_from_state(frozen28)
        with torch.inference_mode(False):
            self.route_hidden = nn.Linear(768,16,device='cpu',dtype=torch.float32)
            self.route_output = nn.Linear(16,2,device='cpu',dtype=torch.float32)
        self.eval().requires_grad_(False)

    def route_features(self, mentions):
        """A context-independent surface terminal and both actual raw contexts."""
        validate_model(self, learning=self.route_hidden.weight.requires_grad)
        with torch.inference_mode(False), torch.no_grad():
            batch = native.encode_mentions(mentions)
            states, _ = self.canonicalizer.encode_batch(batch)
            left,right,_ = self.canonicalizer._contexts(batch)
            terminal = states[torch.arange(len(mentions)),batch.input_lengths]
            features = torch.cat((terminal,left,right),-1).detach().clone().contiguous()
        validate_features(features)
        return features

    def route_logits(self, features):
        validate_features(features)
        return self.route_output(torch.tanh(self.route_hidden(features)))

    @torch.no_grad()
    def decode_with_routes(self, mentions):
        validate_model(self, learning=False)
        features = self.route_features(mentions)
        logits = self.route_logits(features)
        routes = logits.argmax(-1).tolist()
        selected=[]
        for row,route in zip(mentions,routes,strict=True):
            if route==FULL: selected.append(row)
            else:
                surface=row.text[row.char_start:row.char_end]
                selected.append(native.ContextMention(' '+surface,1,1+len(surface)))
        # The whole context intervention precedes the unchanged native model.
        # Exactly one selected decode path runs; no two-prefix score mixing.
        decoded=self.canonicalizer.decode_mentions(tuple(selected))
        return decoded,tuple(routes),logits.detach().clone()

    def decode_mentions(self, mentions):
        return self.decode_with_routes(mentions)[0]

def validate_features(features):
    require(type(features) is torch.Tensor and features.ndim==2 and features.shape[1]==768
        and len(features)>0 and features.dtype==torch.float32 and features.device.type=='cpu'
        and not features.requires_grad and not features.is_inference() and features.is_contiguous()
        and bool(torch.isfinite(features).all()), 'ordinary detached N768 frozen raw features required')

def complete_state(model):
    return {**dict(model.canonicalizer.state_dict()),
        **{k:v.detach() for k,v in model.named_parameters() if k in ROUTE_SHAPES}}

def validate_model(model, *, learning):
    require(type(model) is RoutedCAN32 and type(learning) is bool, 'exact routed32 class/partition required')
    require(not any(callable(getattr(type(model),k,None)) for k in vars(model)), 'immutable native router methods required')
    native.validate_frozen(model.canonicalizer)
    state=complete_state(model);validate_state(state)
    params={k:v for k,v in model.named_parameters() if k in ROUTE_SHAPES}
    require(set(params)==set(ROUTE_SHAPES) and not list(model.buffers())
        and all(not m.training for m in model.modules())
        and all(v.requires_grad is learning and (v.grad is None or learning and v.grad.dtype==torch.float32
            and v.grad.device.type=='cpu' and bool(torch.isfinite(v.grad).all())) for v in params.values()),
        'only router4 may learn; frozen28/no buffers/all eval required')
    require(sum(v.numel() for v in model.parameters())==PARAMETER_COUNT
        and sum(v.numel() for v in params.values())==TRAINABLE_PARAMETERS,
        'honest32/870332 total and4/12338 active inventory required')
    require(all(getattr(m,'_compiled_call_impl',None) is None and not any(getattr(m,k,None)
        for k in ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks'))
        for m in model.modules()), 'no numeric hooks or compiled substitutes required')

def model_from_state(state, *, learning=False):
    validate_state(state)
    with torch.random.fork_rng(devices=[]), torch.inference_mode(False):
        model=RoutedCAN32({k:state[k] for k in native.STATE_SHAPES})
        with torch.no_grad():
            for k,p in model.named_parameters():
                if k in ROUTE_SHAPES: p.copy_(state[k])
    for k,p in model.named_parameters():
        if k in ROUTE_SHAPES: p.requires_grad_(learning)
    validate_model(model,learning=learning)
    actual=complete_state(model)
    require(state_digest(actual)==state_digest(state)
        and all(actual[k].untyped_storage().data_ptr()!=v.untyped_storage().data_ptr() for k,v in state.items()),
        'strict independent ordinary32 restore required')
    return model
