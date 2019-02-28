# -*- coding: utf-8 -*-
#!/usr/bin/env python2
'''
Based on code by kyubyong park at https://www.github.com/kyubyong/dc_tts
'''

from data_load import get_batch, load_vocab
from networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN
import tensorflow as tf
from utils import get_global_attention_guide, learning_rate_decay

class Graph(object):

    def __init__(self, hp, mode="train", reuse=None):
        assert mode in ['train', 'synthesize', 'synthesize_non_monotonic'] 
        self.mode = mode
        self.training = True if mode=="train" else False
        self.reuse = reuse
        self.hp = hp

        self.add_data(reuse=reuse)                     ## TODO: reuse?? 
        self.build_model()
        if self.training:
            self.build_loss()
            self.build_training_scheme()
            

    def add_data(self, reuse=None):
        '''
        Add either variables (for training) or placeholders (for synthesis) to the graph
        '''
        # Data Feeding
        ## L: Text. (B, N), int32
        ## mels: Reduced melspectrogram. (B, T/r, n_mels) float32
        ## mags: Magnitude. (B, T, n_fft//2+1) float32
        hp = self.hp
        if self.mode is 'train':
            batchdict = get_batch(hp, self.get_batchsize(), get_speaker_codes=hp.multispeaker, n_utts=hp.n_utts)

            if 0: print (batchdict) ; print (batchdict.keys()) ; sys.exit('vsfbd')

            self.L, self.mels, self.mags, self.fnames, self.num_batch = \
                batchdict['text'], batchdict['mel'], batchdict['mag'], batchdict['fname'], batchdict['num_batch'] 

            if hp.multispeaker:
                ## check multispeaker config is valid:-
                for position in hp.multispeaker:
                    assert position in ['text_encoder_input', 'text_encoder_towards_end', \
                                'audio_decoder_input', 'ssrn_input', 'audio_encoder_input']
                self.speakers = batchdict['speaker'] 
            else:
                self.speakers = None
            if hp.attention_guide_dir:
                self.gts = batchdict['attention_guide']
            else:
                self.gts = tf.convert_to_tensor(get_global_attention_guide(hp))
            batchsize = self.get_batchsize()
            self.prev_max_attentions = tf.ones(shape=(batchsize,), dtype=tf.int32)
            
        elif self.mode is 'synthesize':  # synthesis
            self.L = tf.placeholder(tf.int32, shape=(None, None))
            self.speakers = None
            if hp.multispeaker:
                self.speakers = tf.placeholder(tf.int32, shape=(None, None))
            self.mels = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels))
            self.prev_max_attentions = tf.placeholder(tf.int32, shape=(None,))
        elif self.mode is 'synthesize_non_monotonic':
            self.L = tf.placeholder(tf.int32, shape=(None, None))
            self.speakers = None
            if hp.multispeaker:
                self.speakers = tf.placeholder(tf.int32, shape=(None, None))
            self.mels = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels))
            self.prev_max_attentions = tf.placeholder(tf.int32, shape=(None,))

            

    def build_training_scheme(self):
        hp = self.hp
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = learning_rate_decay(hp.lr, self.global_step)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        tf.summary.scalar("lr", self.lr)

        ## gradient clipping
        self.gvs = self.optimizer.compute_gradients(self.loss)
        self.clipped = []
        for grad, var in self.gvs:
            grad = tf.clip_by_value(grad, -1., 1.)
            self.clipped.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

        # Summary
        self.merged = tf.summary.merge_all()


class SSRNGraph(Graph):

    def get_batchsize(self):
        return self.hp.batchsize['ssrn']   ## TODO: naming?

    def build_model(self):
        with tf.variable_scope("SSRN"):
            ## OSW: use 'mels' for input both in training and synthesis -- can be either variable or placeholder 
            self.Z_logits, self.Z = SSRN(self.hp, self.mels, training=self.training, speaker_codes=self.speakers, reuse=self.reuse)

    def build_loss(self):
        # mag L1 loss
        self.loss_mags = tf.reduce_mean(tf.abs(self.Z - self.mags))

        # mag binary divergence loss
        self.loss_bd2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Z_logits, labels=self.mags))

        # total loss
        self.lw_mag = self.hp.lw_mag
        self.lw_bd2 = self.hp.lw_bd2                    
        self.loss = (self.lw_mag * self.loss_mags) + (self.lw_bd2 * self.loss_bd2)

        # loss_components attribute is used for reporting to log (osw)
        self.loss_components = [self.loss, self.loss_mags, self.loss_bd2]

        # summary used for reporting to tensorboard (kp)
        tf.summary.scalar('train/loss_mags', self.loss_mags)
        tf.summary.scalar('train/loss_bd2', self.loss_bd2)
        tf.summary.image('train/mag_gt', tf.expand_dims(tf.transpose(self.mags[:1], [0, 2, 1]), -1))
        tf.summary.image('train/mag_hat', tf.expand_dims(tf.transpose(self.Z[:1], [0, 2, 1]), -1))



class Text2MelGraph(Graph):

    def get_batchsize(self):
        return self.hp.batchsize['t2m'] ## TODO: naming?

    def build_model(self):
        with tf.variable_scope("Text2Mel"):
            # Get S or decoder inputs. (B, T//r, n_mels). This is audio shifted 1 frame to the right.
            self.S = tf.concat((tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1)

            # Networks
            with tf.variable_scope("TextEnc"):
                self.K, self.V = TextEnc(self.hp, self.L, training=self.training, speaker_codes=self.speakers, reuse=self.reuse)  # (N, Tx, e)

            with tf.variable_scope("AudioEnc"):
                self.Q = AudioEnc(self.hp, self.S, training=self.training, speaker_codes=self.speakers, reuse=self.reuse)

            with tf.variable_scope("Attention"):
                # R: (B, T/r, 2d)
                # alignments: (B, N, T/r)
                # max_attentions: (B,)
                if self.mode is 'synthesize':
                    self.R, self.alignments, self.max_attentions = Attention(self.hp, self.Q, self.K, self.V,
                                                                            monotonic_attention=True,
                                                                            prev_max_attentions=self.prev_max_attentions)
                elif self.mode is 'train': 
                    self.R, self.alignments, self.max_attentions = Attention(self.hp, self.Q, self.K, self.V,
                                                                            monotonic_attention=False,
                                                                            prev_max_attentions=self.prev_max_attentions)
                elif self.mode is 'synthesize_non_monotonic': 
                    self.R, self.alignments, self.max_attentions = Attention(self.hp, self.Q, self.K, self.V,
                                                                            monotonic_attention=False,
                                                                            prev_max_attentions=None)

            with tf.variable_scope("AudioDec"):
                self.Y_logits, self.Y = AudioDec(self.hp, self.R, training=self.training, speaker_codes=self.speakers, reuse=self.reuse) # (B, T/r, n_mels)

    def build_loss(self):
        hp = self.hp
        # mel L1 loss
        self.loss_mels = tf.reduce_mean(tf.abs(self.Y - self.mels))

        # mel binary divergence loss
        self.loss_bd1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Y_logits, labels=self.mels))

        # guided_attention loss
        self.A = tf.pad(self.alignments, [(0, 0), (0, hp.max_N), (0, hp.max_T)], mode="CONSTANT", constant_values=-1.)[:, :hp.max_N, :hp.max_T]
        if hp.attention_guide_dir:
            self.gts = tf.pad(self.gts, [(0, 0), (0, hp.max_N), (0, hp.max_T)], mode="CONSTANT", constant_values=1.0)[:, :hp.max_N, :hp.max_T] ## TODO: check adding penalty here (1.0 is the right thing)               
        self.attention_masks = tf.to_float(tf.not_equal(self.A, -1))
        self.loss_att = tf.reduce_sum(tf.abs(self.A * self.gts) * self.attention_masks)    ## (B, Letters, Frames) * (Letters, Frames) -- Broadcasting first adds singleton dimensions to the left until rank is matched. 
        self.mask_sum = tf.reduce_sum(self.attention_masks)
        self.loss_att /= self.mask_sum

        # total loss

        ## loss weights
        self.lw_mel = hp.lw_mel
        self.lw_bd1 = hp.lw_bd1
        self.lw_att = hp.lw_att                                            
        
        self.loss = (self.lw_mel * self.loss_mels) + (self.lw_bd1 * self.loss_bd1) + (self.lw_att * self.loss_att)

        # loss_components attribute is used for reporting to log (osw)
        self.loss_components = [self.loss, self.loss_mels, self.loss_bd1, self.loss_att]

        # summary used for reporting to tensorboard (kp)
        tf.summary.scalar('train/loss_mels', self.loss_mels)
        tf.summary.scalar('train/loss_bd1', self.loss_bd1)
        tf.summary.scalar('train/loss_att', self.loss_att)
        tf.summary.image('train/mel_gt', tf.expand_dims(tf.transpose(self.mels[:1], [0, 2, 1]), -1))
        tf.summary.image('train/mel_hat', tf.expand_dims(tf.transpose(self.Y[:1], [0, 2, 1]), -1))




class TextEncGraph(Graph):  ## partial graph for deployment only

    def build_model(self):
        with tf.variable_scope("Text2Mel"):
            # Get S or decoder inputs. (B, T//r, n_mels)
            self.S = tf.concat((tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1)

            # Networks
            with tf.variable_scope("TextEnc"):
                self.K, self.V = TextEnc(self.hp, self.L, training=self.training, speaker_codes=self.speakers, reuse=self.reuse)  # (N, Tx, e)



class BabblerGraph(Graph):
    '''
    A model which simply predicts the next audio step given an audio history. Can be used
    by itself to babble at synthesis time, given some initial seed (e.g. some frames of
    silence, or the beginning of a sentence to be completed). Alternatively, its weights can 
    be used to initialise the corresponding weights of a text2mel model. As in the paper
    "Semi-Supervised Training for Improving Data Efficiency in End-to-End Speech Synthesis" by
    Yu-An Chung et al. (2018: https://arxiv.org/abs/1808.10128), dummy textencoder outputs 
    consisting of all zeros are supplied in training.
    '''
    def get_batchsize(self):
        return self.hp.batchsize.get('babbler', 32) ## default = 32 

    def build_model(self):
        with tf.variable_scope("Text2Mel"): ## keep scope names consistent with full Text2Mel 
                                            ## to allow parameters to be reused more easily later
            # Get S or decoder inputs. (B, T//r, n_mels). This is audio shifted 1 frame to the right.
            self.S = tf.concat((tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1)

            ## Babbler has no TextEnc

            with tf.variable_scope("AudioEnc"):
                self.Q = AudioEnc(self.hp, self.S, training=self.training, reuse=self.reuse)

            with tf.variable_scope("Attention"):
                ## Babbler has no real attention. Dummy (all 0) text encoder outputs are supplied instead.
                # R: concat Q with zero vector (dummy text encoder outputs)
                dummy_R_prime = tf.zeros_like(self.Q) ## R_prime shares shape of audio encoder output
                self.R = tf.concat((dummy_R_prime, self.Q), -1) 

            with tf.variable_scope("AudioDec"):
                self.Y_logits, self.Y = AudioDec(self.hp, self.R, training=self.training, speaker_codes=self.speakers, reuse=self.reuse) # (B, T/r, n_mels)


    def build_loss(self):
        hp = self.hp
        # mel L1 loss
        self.loss_mels = tf.reduce_mean(tf.abs(self.Y - self.mels))
        # mel binary divergence loss
        self.loss_bd = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Y_logits, labels=self.mels))

        # total loss, with 2 terms combined with loss weights:
        self.loss = (hp.loss_weights['babbler']['L1'] * self.loss_mels) + \
                    (hp.loss_weights['babbler']['binary_divergence'] * self.loss_bd) 

        # loss_components attribute is used for reporting to log (osw)
        self.loss_components = [self.loss, self.loss_mels, self.loss_bd]

        # summary used for reporting to tensorboard (kp)
        tf.summary.scalar('train/loss_mels', self.loss_mels)
        tf.summary.scalar('train/loss_bd', self.loss_bd)
        tf.summary.image('train/mel_gt', tf.expand_dims(tf.transpose(self.mels[:1], [0, 2, 1]), -1))
        tf.summary.image('train/mel_hat', tf.expand_dims(tf.transpose(self.Y[:1], [0, 2, 1]), -1))

