    # -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts

Modified...
'''

from __future__ import print_function

from modules import *
import tensorflow as tf

def TextEnc(hp, L, training=True, speaker_codes=None, reuse=None):
    '''
    Args:
      L: Text inputs. (B, N)

    Return:
        K: Keys. (B, N, d)
        V: Values. (B, N, d)
    '''
    i = 1
    tensor = embed(L,
                   vocab_size=len(hp.vocab),
                   num_units=hp.e,
                   scope="embed_{}".format(i), reuse=reuse); i += 1
    if hp.multispeaker in [1, 2]:
        speaker_codes_time = tf.tile(speaker_codes, [1,tf.shape(L)[1]])
        speaker_reps = embed(speaker_codes_time,
                       vocab_size=hp.nspeakers,
                       num_units=hp.speaker_embedding_size,
                       scope="embed_{}".format(i), reuse=reuse); i += 1 
        tensor = tf.concat((tensor, speaker_reps), -1)
    tensor = conv1d(tensor,
                    filters=2*hp.d,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1

    for outer_counter in range(2):
        for j in range(4):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            dropout_rate=hp.dropout_rate,
                            activation_fn=None,
                            training=training,
                            scope="HC_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1

    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        activation_fn=None,
                        training=training,
                        scope="HC_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1

    for _ in range(2):
        tensor = hc(tensor,
                        size=1,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        activation_fn=None,
                        training=training,
                        scope="HC_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1

    K, V = tf.split(tensor, 2, -1)
    return K, V

def AudioEnc(hp, S, training=True, speaker_codes=None, reuse=None):
    '''
    Args:
      S: melspectrogram. (B, T/r, n_mels)

    Returns
      Q: Queries. (B, T/r, d)
    '''
    i = 1
    tensor = conv1d(S,
                    filters=hp.d,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1

    if hp.multispeaker in [2]:
        speaker_codes_time = tf.tile(speaker_codes, [1,tf.shape(S)[1]])   
        speaker_reps = embed(speaker_codes_time,
                       vocab_size=hp.nspeakers,
                       num_units=hp.speaker_embedding_size,
                       scope="embed_{}".format(i), reuse=reuse); i += 1 
        tensor = tf.concat((tensor, speaker_reps), -1)
        tensor = conv1d(tensor, filters=hp.d, size=1, rate=1, dropout_rate=hp.dropout_rate, \
                    training=training, scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1

    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1
    for _ in range(2):
        for j in range(4):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            padding="CAUSAL",
                            dropout_rate=hp.dropout_rate,
                            training=training,
                            scope="HC_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=3,
                        padding="CAUSAL",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1

    return tensor

def Attention(hp, Q, K, V, mononotic_attention=False, prev_max_attentions=None):
    '''
    Args:
      Q: Queries. (B, T/r, d)
      K: Keys. (B, N, d)
      V: Values. (B, N, d)
      mononotic_attention: A boolean. At training, it is False.
      prev_max_attentions: (B,). At training, it is set to None.

    Returns:
      R: [Context Vectors; Q]. (B, T/r, 2d)
      alignments: (B, N, T/r)
      max_attentions: (B, T/r)
    '''
    A = tf.matmul(Q, K, transpose_b=True) * tf.rsqrt(tf.to_float(hp.d))
    if mononotic_attention:  # for inference
        key_masks = tf.sequence_mask(prev_max_attentions, hp.max_N)
        reverse_masks = tf.sequence_mask(hp.max_N - hp.attention_win_size - prev_max_attentions, hp.max_N)[:, ::-1]
        masks = tf.logical_or(key_masks, reverse_masks)
        masks = tf.tile(tf.expand_dims(masks, 1), [1, hp.max_T, 1])
        paddings = tf.ones_like(A) * (-2 ** 32 + 1)  # (B, T/r, N)
        A = tf.where(tf.equal(masks, False), A, paddings)  # where(condition,x,y) --Return the elements, either from x or y, depending on the condition.
    A = tf.nn.softmax(A) # (B, T/r, N)
    max_attentions = tf.argmax(A, -1)  # (B, T/r)
    R = tf.matmul(A, V)
    R = tf.concat((R, Q), -1)

    alignments = tf.transpose(A, [0, 2, 1]) # (B, N, T/r)

    return R, alignments, max_attentions

def AudioDec(hp, R, training=True, speaker_codes=None, reuse=None):
    '''
    Args:
      R: [Context Vectors; Q]. (B, T/r, 2d)

    Returns:
      Y: Melspectrogram predictions. (B, T/r, n_mels)
    '''

    i = 1
    tensor = conv1d(R,
                    filters=hp.d,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1

    if hp.multispeaker in [2, 3]:
        speaker_codes_time = tf.tile(speaker_codes, [1,tf.shape(R)[1]])   
        speaker_reps = embed(speaker_codes_time,    
                       vocab_size=hp.nspeakers,
                       num_units=hp.speaker_embedding_size,
                       scope="embed_{}".format(i), reuse=reuse); i += 1 
        tensor = tf.concat((tensor, speaker_reps), -1)
        tensor = conv1d(tensor, filters=hp.d, size=1, rate=1, dropout_rate=hp.dropout_rate, \
                    training=training, scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1

    for j in range(4):
        tensor = hc(tensor,
                        size=3,
                        rate=3**j,
                        padding="CAUSAL",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1

    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        padding="CAUSAL",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1
    for _ in range(3):
        tensor = conv1d(tensor,
                        size=1,
                        rate=1,
                        padding="CAUSAL",
                        dropout_rate=hp.dropout_rate,
                        activation_fn=tf.nn.relu,
                        training=training,
                        scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1
    # mel_hats
    logits = conv1d(tensor,
                    filters=hp.n_mels,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1
    Y = tf.nn.sigmoid(logits) # mel_hats

    return logits, Y

def SSRN(hp, Y, training=True, speaker_codes=None, reuse=None):
    '''
    Args:
      Y: Melspectrogram Predictions. (B, T/r, n_mels)

    Returns:
      Z: Spectrogram Predictions. (B, T, 1+n_fft/2)
    '''

    i = 1 # number of layers

    # -> (B, T/r, c)
    tensor = conv1d(Y,
                    filters=hp.c,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1

    if hp.multispeaker in [2]:
        speaker_codes_time = tf.tile(speaker_codes, [1,tf.shape(Y)[1]])   
        speaker_reps = embed(speaker_codes_time,            
                       vocab_size=hp.nspeakers,
                       num_units=hp.speaker_embedding_size,
                       scope="embed_{}".format(i), reuse=reuse); i += 1 
        tensor = tf.concat((tensor, speaker_reps), -1)    
        tensor = conv1d(tensor, filters=hp.c, size=1, rate=1, dropout_rate=hp.dropout_rate, \
                    training=training, scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1

    for j in range(2):
        tensor = hc(tensor,
                      size=3,
                      rate=3**j,
                      dropout_rate=hp.dropout_rate,
                      training=training,
                      scope="HC_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1
    if hp.r==4:
        n_transposes=2
    elif hp.r==8:
        n_transposes=3
    else:
        sys.exit('reduction factor not handled by SSRN!')

    for _ in range(n_transposes):
        # -> (B, T/2, c) -> (B, T, c)
        tensor = conv1d_transpose(tensor,
                                  scope="D_{}".format(i),
                                  dropout_rate=hp.dropout_rate,
                                  training=training, reuse=reuse); i += 1
        for j in range(2):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            dropout_rate=hp.dropout_rate,
                            training=training,
                            scope="HC_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1
    # -> (B, T, 2*c)
    tensor = conv1d(tensor,
                    filters=2*hp.c,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1
    # -> (B, T, 1+n_fft/2)

    tensor = conv1d(tensor,
                    filters=hp.full_dim, 
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1

    for _ in range(2):
        tensor = conv1d(tensor,
                        size=1,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        activation_fn=tf.nn.relu,
                        training=training,
                        scope="C_{}".format(i), normtype=hp.norm, reuse=reuse); i += 1
    logits = conv1d(tensor,
               size=1,
               rate=1,
               dropout_rate=hp.dropout_rate,
               training=training,
               scope="C_{}".format(i), normtype=hp.norm, reuse=reuse)
    Z = tf.nn.sigmoid(logits)
    return logits, Z
