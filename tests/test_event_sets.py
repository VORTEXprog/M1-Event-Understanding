"""Exact all-span event-set decisions/gradients on invented score inventories."""
import itertools
import math
import pytest
import torch
from m1 import typed_event_set_localization_loss_v1 as loss


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def bank(widths=(4,),seed=114):
    generator=torch.Generator().manual_seed(seed);po=[0];uo=[0];pairs=[]
    for width in widths:
        pairs.append(loss.complete_pairs(width));po.append(po[-1]+width*(width+1)//2);uo.append(uo[-1]+width)
    return loss.PackedEventSetScores(torch.randn(po[-1],7,generator=generator,requires_grad=True),
        torch.randn(uo[-1],generator=generator,requires_grad=True),torch.cat(pairs),torch.tensor(po),torch.tensor(uo))


def exhaustive(values,row=0):
    ps,pe=int(values.pair_offsets[row]),int(values.pair_offsets[row+1]);us,ue=int(values.sentence_unit_offsets[row]),int(values.sentence_unit_offsets[row+1])
    scores=values.pair_scores.detach()[ps:pe];outside=values.outside_logits.detach()[us:ue];width=len(outside)
    lookup={(int(i),int(j)):index for index,(i,j) in enumerate(values.pair_indices[ps:pe])}
    def visit(start,chosen,value):
        if start==width:yield tuple(chosen),value;return
        yield from visit(start+1,chosen,value+float(outside[start]))
        for stop in range(start+1,width+1):
            for kind in range(7):
                yield from visit(stop,chosen+[(start,stop,kind)],value+float(scores[lookup[start,stop],kind]))
    return list(visit(0,[],0.))


def test_complete_partition_MAP_and_gradients_equal_exhaustive_typed_event_sets():
    values=bank((3,));paths=exhaustive(values);scores=torch.tensor([v for _,v in paths],dtype=torch.float64)
    observed=loss.log_partition(values);assert math.isclose(float(observed.detach()),float(scores.logsumexp(0)),abs_tol=2e-6)
    answer=loss.decode_map(values)[0]
    assert answer['map_score']==pytest.approx(max(v for _,v in paths),abs=2e-6)
    assert (answer['spans'],pytest.approx(answer['map_score'],abs=2e-6)) in paths
    observed.sum().backward();probabilities=scores.softmax(0)
    for index,(i,j) in enumerate(values.pair_indices):
        for kind in range(7):
            mass=sum(float(p) for (chosen,_),p in zip(paths,probabilities) if (int(i),int(j),kind) in chosen)
            assert float(values.pair_scores.grad[index,kind])==pytest.approx(mass,abs=2e-6)
    assert sum(not events for events,_ in paths)==1


def test_exact_overlap_competition_fullwidth_no_gold_count_and_type_ties():
    values=bank((4,));scores=torch.full_like(values.pair_scores,-30)
    lookup={tuple(v.tolist()):i for i,v in enumerate(values.pair_indices)}
    scores[lookup[0,3],0]=10;scores[lookup[0,1],1]=6;scores[lookup[2,3],2]=6
    values=loss.PackedEventSetScores(scores,torch.zeros(4),values.pair_indices,values.pair_offsets,values.sentence_unit_offsets)
    assert loss.decode_map(values)[0]['spans']==((0,1,1),(2,3,2))
    scores[lookup[0,4],3]=50
    assert loss.decode_map(values)[0]['spans']==((0,4,3),)
    scores.zero_();assert loss.decode_map(values)[0]['spans']==()


def test_empty_sets_empty_units_and_ragged_batch_are_defined():
    values=bank((0,3,1));zeros=loss.slice_bank(values,0,1)
    assert loss.log_partition(zeros).tolist()==[0.] and loss.decode_map(zeros)[0]['spans']==()
    assert float(loss.negative_log_likelihood(zeros,[()]).detach())==0
    full=loss.log_partition(values)
    assert all(torch.allclose(full[i:i+1],loss.log_partition(loss.slice_bank(values,i,i+1))) for i in range(3))


def test_global_row_denominator_preserves_unequal_microchunks_gradients():
    values=bank((3,1,4));gold=[((0,1,0),),(),((0,2,6),(2,4,1))]
    full=loss.negative_log_likelihood(values,gold)
    micro=loss.negative_log_likelihood(loss.slice_bank(values,0,2),gold[:2],global_row_count=3)
    micro+=loss.negative_log_likelihood(loss.slice_bank(values,2,3),gold[2:],global_row_count=3)
    assert torch.equal(full,micro) or torch.allclose(full,micro,atol=2e-6,rtol=2e-6)
    a=torch.autograd.grad(full,(values.pair_scores,values.outside_logits),retain_graph=True)
    b=torch.autograd.grad(micro,(values.pair_scores,values.outside_logits))
    assert all(torch.allclose(x,y,atol=2e-6,rtol=2e-6) for x,y in zip(a,b))
    with pytest.raises(ValueError):loss.negative_log_likelihood(values,gold,global_row_count=2)


@pytest.mark.parametrize('gold',(((0,2,0),(1,3,1)),((0,4,0),),((0,1,7),),((0.0,1,0),),((0,1,False),)))
def test_wrong_gold_geometry_type_or_python_numeric_taint_rejected(gold):
    with pytest.raises(ValueError):loss.gold_score(bank((3,)),[gold])


@pytest.mark.parametrize('defect',('missing_pair','duplicate_pair','missing_type','nan','offset','float_geometry','extra_key'))
def test_complete_candidate_bank_rejects_pruning_geometry_and_schema(defect):
    values=bank((3,));fields=values.as_tensor_dict()
    if defect=='missing_pair':fields['pair_indices']=fields['pair_indices'][:-1]
    if defect=='duplicate_pair':fields['pair_indices']=fields['pair_indices'].clone();fields['pair_indices'][-1]=fields['pair_indices'][0]
    if defect=='missing_type':fields['pair_scores']=fields['pair_scores'][:,:6].contiguous()
    if defect=='nan':fields['outside_logits']=fields['outside_logits'].clone();fields['outside_logits'][0]=float('nan')
    if defect=='offset':fields['pair_offsets']=torch.tensor([0,5])
    if defect=='float_geometry':fields['pair_indices']=fields['pair_indices'].float()
    if defect=='extra_key':fields['gold_event_count']=torch.tensor([1])
    with pytest.raises(ValueError):loss.bank_from_tensors(fields)


def test_wrong_event_type_does_not_disappear_in_span_projection():
    gold=((0,1,0),);wrong=((0,1,1),)
    assert {(i,j) for i,j,k in gold}=={(i,j) for i,j,k in wrong} and gold!=wrong
