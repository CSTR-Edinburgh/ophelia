
'''
TODO: logSpecDbDist appropriate? (both mels & mags?)
TODO: compute output length error?
TODO: work out best way of handling the fact that predicted *coarse* features 
      can correspond to text but be arbitrarily 'out of phase' with reference.
      Mutliple references? Or compare against full-time resolution reference? 
'''
import logging
from mcd import dtw
import mcd.metrics_fast as mt
def compute_dtw_error(reference, predictions):
    minCostTot = 0.0
    framesTot = 0
    for (nat, synth) in zip(reference, predictions):
        nat, synth = nat.astype('float64'), synth.astype('float64')
        minCost, path = dtw.dtw(nat, synth, mt.logSpecDbDist)
        frames = len(nat)
        minCostTot += minCost
        framesTot += frames
        #print ('LSD = %f (%d/%s frames nat/synth)' % (minCost / frames, frames, len(synth)))
    mean_score = minCostTot / framesTot
    print ('overall LSD = %f (%s frames nat/synth)' % (mean_score, framesTot))
    return mean_score

def compute_simple_LSD(reference_list, prediction_list):
    costTot = 0.0
    framesTot = 0    
    for (synth, nat) in zip(prediction_list, reference_list):
        #synth = prediction_tensor[i,:,:].astype('float64')
        # len_nat = len(nat)
        assert len(synth) == len(nat)
        #synth = synth[:len_nat, :]
        nat = nat.astype('float64')
        synth = synth.astype('float64')
        cost = sum([
            mt.logSpecDbDist(natFrame, synthFrame)
            for natFrame, synthFrame in zip(nat, synth)
        ])
        #logging.debug('Sentence LSD: %s'%(cost))
        framesTot += len(nat)
        costTot += cost
    #print 'overall MCD = %f (%d frames)' % (costTot / framesTot, framesTot)
    return costTot / framesTot
    

