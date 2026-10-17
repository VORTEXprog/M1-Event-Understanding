"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import torch

VERSION='M1_EXACT_ALL_SPAN_SEVEN_TYPE_EVENT_SET_SEMIMARKOV_v1'

SOURCE=Path('src/m1/typed_event_set_localization_loss_v1.py')

TYPE_ORDER=('PURCHASE','GIVE','OWN','OPEN','CLOSE','REPORT','SHOW')

OBJECTIVE='SUM_EXACT_TYPED_EVENT_SET_ROW_NLL_DIVIDED_BY_GLOBAL_ACTUAL_ROW_COUNT'

DECODER='EXACT_SEMIMARKOV_MAP_NONE_THEN_EARLIEST_START_THEN_NATIVE_TYPE_TIES_v1'

BANK_KEYS=('pair_scores','outside_logits','pair_indices','pair_offsets','sentence_unit_offsets')

@dataclass(frozen=True)
class PackedEventSetScores:
    pair_scores:torch.Tensor
    outside_logits:torch.Tensor
    pair_indices:torch.Tensor
    pair_offsets:torch.Tensor
    sentence_unit_offsets:torch.Tensor

    def as_tensor_dict(self):return {name:getattr(self,name) for name in BANK_KEYS}

    def detach(self):
        return PackedEventSetScores(**{name:value.detach().clone().contiguous()
            for name,value in self.as_tensor_dict().items()})

def complete_pairs(width):
    if type(width) is not int or width<0:raise ValueError('nonnegative exact unit count required')
    pairs=torch.triu_indices(width,width,device='cpu').t().contiguous()
    if len(pairs):pairs[:,1]+=1
    return pairs

def validate_bank(bank):
    if type(bank) is not PackedEventSetScores:raise ValueError('exact packed all-span score class required')
    values=bank.as_tensor_dict()
    for name,value in values.items():
        if (not isinstance(value,torch.Tensor) or value.device.type!='cpu'
                or value.dtype!=(torch.float32 if name in BANK_KEYS[:2] else torch.int64)
                or not value.is_contiguous() or not bool(value.isfinite().all())
                or (name not in BANK_KEYS[:2] and value.requires_grad)):
            raise ValueError('finite contiguous CPU FP32 scores and int64 geometry required')
    scores,outside,pairs,po,uo=(values[name] for name in BANK_KEYS)
    if (scores.ndim!=2 or scores.shape[1]!=7 or outside.ndim!=1 or pairs.shape!=(len(scores),2)
            or po.ndim!=1 or uo.ndim!=1 or not len(uo) or len(po)!=len(uo)
            or po[0].item()!=0 or uo[0].item()!=0 or po[-1].item()!=len(scores)
            or uo[-1].item()!=len(outside) or not bool((po[1:]>=po[:-1]).all())
            or not bool((uo[1:]>=uo[:-1]).all())):
        raise ValueError('complete ragged pair/unit offset inventory required')
    for row in range(len(uo)-1):
        width=int(uo[row+1]-uo[row]);start,stop=int(po[row]),int(po[row+1])
        if not torch.equal(pairs[start:stop],complete_pairs(width)):
            raise ValueError('all valid intervals exactly once in start-major order required')
    return bank

def bank_from_tensors(values):
    if type(values) is not dict or set(values)!=set(BANK_KEYS):raise ValueError('exact saved score bank keys required')
    return validate_bank(PackedEventSetScores(**values))

def _row(bank,row):
    ps,pe=int(bank.pair_offsets[row]),int(bank.pair_offsets[row+1])
    us,ue=int(bank.sentence_unit_offsets[row]),int(bank.sentence_unit_offsets[row+1])
    return bank.pair_scores[ps:pe],bank.outside_logits[us:ue],bank.pair_indices[ps:pe]

def log_partition(bank):
    validate_bank(bank)
    rows=len(bank.pair_offsets)-1
    if not rows:return bank.outside_logits.new_empty((0,))
    width=int((bank.sentence_unit_offsets[1:]-bank.sentence_unit_offsets[:-1]).max())
    intervals=bank.pair_scores.new_full((rows,width,width),-torch.inf)
    outside=bank.outside_logits.new_zeros((rows,width))
    for row in range(len(bank.pair_offsets)-1):
        scores,local,pairs=_row(bank,row)
        # Summing type alternatives is valid because the model is zeroth-order:
        # it has sentence-conditioned potentials but no cross-event transition.
        if len(pairs):intervals[row,pairs[:,0],pairs[:,1]-1]=scores.logsumexp(-1)
        outside[row,:len(local)]=local
    # Padded cells have no event edges and their sole NONE step is neutral.
    alpha=[outside.new_zeros(rows)]
    for end in range(1,width+1):
        event=torch.stack(alpha,dim=1)+intervals[:,:end,end-1]
        stay=alpha[-1]+outside[:,end-1]
        alpha.append(torch.logsumexp(torch.cat((stay[:,None],event),dim=1),dim=1))
    return alpha[-1]

def _validate_gold(gold,width):
    if type(gold) not in (list,tuple):raise ValueError('one loss-only ordered event set required')
    previous=0;result=[]
    for span in gold:
        if (type(span) not in (list,tuple) or len(span)!=3 or any(type(v) is not int for v in span)
                or not previous<=span[0]<span[1]<=width or not 0<=span[2]<7):
            raise ValueError('gold must be sorted disjoint unit-exact native typed spans')
        result.append(tuple(span));previous=span[1]
    return tuple(result)

def gold_score(bank,gold_sets):
    validate_bank(bank)
    if type(gold_sets) not in (tuple,list) or len(gold_sets)!=len(bank.pair_offsets)-1:
        raise ValueError('one gold set per authentic row required')
    values=[]
    for row,gold in enumerate(gold_sets):
        scores,outside,pairs=_row(bank,row);gold=_validate_gold(gold,len(outside))
        used=torch.zeros(len(outside),dtype=torch.bool);chosen=[]
        for start,end,kind in gold:
            # Closed-form start-major index, no annotation-derived candidate list.
            index=start*len(outside)-start*(start-1)//2+(end-start-1)
            chosen.append(scores[index,kind]);used[start:end]=True
        values.append(outside[~used].sum()+sum(chosen,outside.new_zeros(())))
    return torch.stack(values) if values else bank.outside_logits.new_empty((0,))

def negative_log_likelihood(bank,gold_sets,*,global_row_count=None,reduction='mean'):
    """Micro SUM / original global count; one optimizer step belongs to caller."""
    if reduction not in ('sum','mean'):raise ValueError('sum or global-row mean required')
    terms=log_partition(bank)-gold_score(bank,gold_sets)
    if reduction=='sum':
        if global_row_count is not None:raise ValueError('SUM has no hidden denominator')
        return terms.sum()
    denominator=len(terms) if global_row_count is None else global_row_count
    if type(denominator) is not int or denominator<=0 or denominator<len(terms):
        raise ValueError('original positive global actual row count required')
    return terms.sum()/denominator

def decode_map(bank):
    validate_bank(bank);result=[]
    # Pure score-space decisions; tensors are detached, no neural constructor.
    with torch.no_grad():
        for row in range(len(bank.pair_offsets)-1):
            scores,outside,pairs=_row(bank,row);best=[0.];back=[]
            for end in range(1,len(outside)+1):
                value=best[-1]+float(outside[end-1]);edge=(end-1,None)
                for index in (pairs[:,1]==end).nonzero().flatten().tolist():
                    start=int(pairs[index,0]);score,kind=scores[index].max(0)
                    candidate=best[start]+float(score)
                    if candidate>value:value=candidate;edge=(start,int(kind))
                best.append(value);back.append(edge)
            spans=[];end=len(outside)
            while end:
                start,kind=back[end-1]
                if kind is not None:spans.append((start,end,kind))
                end=start
            result.append({'spans':tuple(reversed(spans)),'map_score':best[-1]})
    return tuple(result)

def slice_bank(bank,start,stop):
    validate_bank(bank)
    if type(start) is not int or type(stop) is not int or not 0<=start<stop<len(bank.pair_offsets):
        raise ValueError('nonempty row slice required')
    a,b=int(bank.pair_offsets[start]),int(bank.pair_offsets[stop]);u,v=int(bank.sentence_unit_offsets[start]),int(bank.sentence_unit_offsets[stop])
    return PackedEventSetScores(bank.pair_scores[a:b].contiguous(),bank.outside_logits[u:v].contiguous(),
        bank.pair_indices[a:b].contiguous(),(bank.pair_offsets[start:stop+1]-a).contiguous(),
        (bank.sentence_unit_offsets[start:stop+1]-u).contiguous())

def gold_event_sets(rows):
    """Loss-side authentic annotations; never a runtime candidate/type source."""
    from .localizer_typed_eventhood_targets_v1 import typed_targets
    labels,_=typed_targets(rows,['AUTHENTICATED_MATRIX_FIT']*len(rows),require_all_types=False,fit_only=False)
    result=[];offset=0
    for row in rows:
        count=len(row['canonical_units']);local=labels[offset:offset+count].tolist();offset+=count
        spans=[];index=0
        while index<count:
            label=local[index]
            if label==0:index+=1;continue
            if not 1<=label<=7:raise ValueError('authentic typed gold begins with B')
            end=index+1
            while end<count and local[end]==label+7:end+=1
            spans.append((index,end,label-1));index=end
        result.append(tuple(spans))
    return tuple(result)
