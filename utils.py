# -*- coding: utf-8 -*-
#!/usr/bin/env python2
'''
Adapted from original code by kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function, division

import math
import sys
import librosa
import os, copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import tensorflow as tf


def get_spectrograms(hp, fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.

    Args:
      fpath: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    try:
        y, sr = librosa.load(fpath, sr=hp.sr)
    except ValueError:
        print('The sound file {} seems to be too short'.format(fpath))
        return None, None

    # Trimming
    if hp.trim_before_spectrogram_extraction:
        y, _ = librosa.effects.trim(y, top_db=hp.trim_before_spectrogram_extraction) #### osw: don't trim here so length matches e.g. magphase features
    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

def spectrogram2wav(hp, mag, trim_output=False):
    '''# Generate wave file from linear magnitude spectrogram

    Args:
      mag: A numpy array of (T, 1+n_fft//2)

    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(hp, mag**hp.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    if trim_output: # removed this as default, as we now do early stopping in generation
        wav, _ = librosa.effects.trim(wav)
    
    return wav.astype(np.float32)

def griffin_lim(hp, spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(hp, X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(hp, X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(hp, spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

# TODO add functionality so that we can also plot on phone identities to the encoder states on the y-axis
def plot_alignment(hp, alignment, utt_idx, t2m_epoch, dir=''):
    """Plots the alignment.

    Args:
      hp: Hyperparams file
      alignment: A numpy array with shape of (encoder_steps, decoder_steps)
      utt_idx: The index of the utterance that we are plotting, for naming/titling purposes.
      t2m_epoch: (int) training epoch reached for text2mel model.
      dir: Output path.
    """
    if not dir:
        dir = hp.logdir
    if not os.path.exists(dir): os.mkdir(dir)

    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    fig.colorbar(im)
    plt.title('Cfg={}, t2m_epoch={}, utt=#{}'.format(
        hp.config_name, t2m_epoch, utt_idx))
    plt.ylabel('Encoder timestep')
    plt.xlabel('Decoder timestep')

    plt.savefig('{}/alignment_{}_utt{}_epoch{}.png'.format(dir,
                                                           hp.config_name, utt_idx, t2m_epoch), format='png')
    plt.close(fig)

def get_attention_guide(xdim, ydim, g=0.2):
    '''Guided attention. Refer to page 3 on the paper.'''
    W = np.zeros((xdim, ydim), dtype=np.float32)
    for n_pos in range(xdim):
        for t_pos in range(ydim):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(ydim) - n_pos / float(xdim)) ** 2 / (2 * g * g))
    return W

def get_global_attention_guide(hp):
    return get_attention_guide(hp.max_N, hp.max_T, g=hp.g)


def learning_rate_decay(init_lr, global_step, warmup_steps = 4000.0):
    '''Noam scheme from tensor2tensor'''
    step = tf.to_float(global_step + 1)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)

def load_spectrograms(hp, fpath):
    '''Read the wave file in `fpath`
    and extracts spectrograms'''

    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(hp, fpath)
    t = mel.shape[0]

    # Marginal padding for reduction shape sync.  TODO: could refactor with end_pad_for_reduction_shape_sync function?
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")

    # Reduction
    mel_reduced = mel[::hp.r, :]
    return fname, mel_reduced, mag, mel


def end_pad_for_reduction_shape_sync(data, hp):
    nframe = data.shape[0]
    num_paddings = hp.r - (nframe % hp.r) if nframe % hp.r != 0 else 0
    data = np.pad(data, [[0, num_paddings], [0, 0]], mode="constant")
    return data


def durations_to_hard_attention_matrix(durations):   
    '''
    Take array of durations, return selection matrix to replace A in attention mechanism.
    E.g.:

    durations_to_hard_attention_matrix(np.array([3,0,1,2]))

    [[1. 1. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 1. 1.]]  
    '''
    nphones = len(durations)
    nframes = durations.sum()
    A = np.zeros((nframes, nphones), dtype=np.float32)
    start = 0
    for (i,dur) in enumerate(durations):
        end = start + dur
        A[start:end,i] = 1.0
        start = end
    assert A.sum(axis=1).all() == 1.0 
    assert A.sum(axis=0).all() == durations.all()
    return A


def durations_to_position(durations, fractional=False):
    
    nframes = durations.sum()
    positions = np.zeros((nframes,), dtype=np.float32)
    start = 0
    #print (durations)
    #sys.exit('qevfewrb')
    for dur in durations:
        #print (positions)
        end = start + dur
        if fractional:
            #print (dur)
            positions[start:end] = np.arange(dur) / dur
        else:
            positions[start:end] = np.arange(dur)
        start = end
    return positions.reshape(-1,1)



def split_streams(combined, streamlist, streamdims):
    separate_streams = {}
    start = 0
    for (stream, dim) in zip(streamlist, streamdims):
        end = start + dim
        stream_speech = combined[:, start:end]
        start = end
        separate_streams[stream] = stream_speech
    return separate_streams


def magphase_synth_from_compressed(split_predictions, samplerate=48000, b_const_rate=5.0):

    required_streams = ['real','imag','lf0','vuv','mag']
    for stream in required_streams:
        assert stream in split_predictions, 'Missing stream: %s'%(stream)

    lfz = split_predictions['lf0'].flatten()
    vuv = split_predictions['vuv'].flatten()

    ## TODO: configure this...
    unvoiced = vuv<0.5
    lfz = np.clip(lfz, math.log(60.0), math.log(400.0))
    lfz[unvoiced] = -10000000000.0

    synwave = mp.synthesis_from_compressed(split_predictions['mag'], split_predictions['real'], \
                    split_predictions['imag'], lfz, samplerate, b_const_rate=b_const_rate) # fft_len=2048,
    
    return synwave

# from: https://nolanbconaway.github.io/blog/2017/softmax-numpy
def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


if __name__ == '__main__':
    if 0:
        import pylab
        a = guided_attention(g=0.2)    
        pylab.imshow(a)
        pylab.show()
    if 1:
        a = durations_to_hard_attention_matrix(np.array([3,0,1,2]))
        print( a)
        print (durations_to_fractional_position(np.array([3,0,1,2])))
        print (durations_to_absolute_position(np.array([3,0,1,2])))
