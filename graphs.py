from data_load import get_batch, load_vocab
from modules import *
from networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN
import tensorflow as tf
from utils import *

class Graph:
    def __init__(self, hp, num=1, mode="train"):
        '''
        Args:
          num: Either 1 or 2. 1 for Text2Mel 2 for SSRN.
          mode: Either "train" or "synthesize".
        '''
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab(hp)

        # Set flag
        training = True if mode=="train" else False

        # Graph
        # Data Feeding
        ## L: Text. (B, N), int32
        ## mels: Reduced melspectrogram. (B, T/r, n_mels) float32
        ## mags: Magnitude. (B, T, n_fft//2+1) float32
        if mode=="train":
            self.L, self.mels, self.mags, self.fnames, self.num_batch = get_batch(hp)
            if 1:
                print('Got batch:')
                print(self.L)
                print(self.mels)
                print(self.mags)
                print(self.fnames)
                print(self.num_batch)

                # Tensor("bucket_by_sequence_length/bucket/dequeue_top:2", shape=(32, ?), dtype=int32, device=/device:CPU:0)
                # Tensor("bucket_by_sequence_length/bucket/dequeue_top:3", shape=(32, ?, 62), dtype=float32, device=/device:CPU:0)
                # Tensor("bucket_by_sequence_length/bucket/dequeue_top:4", shape=(32, ?, 1025), dtype=float32, device=/device:CPU:0)
                # Tensor("bucket_by_sequence_length/bucket/dequeue_top:5", shape=(32,), dtype=string, device=/device:CPU:0)

            self.prev_max_attentions = tf.ones(shape=(hp.B,), dtype=tf.int32)
            self.gts = tf.convert_to_tensor(guided_attention(hp))
        else:  # Synthesize
            self.L = tf.placeholder(tf.int32, shape=(None, None))
            self.mels = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels))
            self.prev_max_attentions = tf.placeholder(tf.int32, shape=(None,))

        if num==1 or (not training):
            with tf.variable_scope("Text2Mel"):
                # Get S or decoder inputs. (B, T//r, n_mels)
                self.S = tf.concat((tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1)

                # Networks
                with tf.variable_scope("TextEnc"):
                    self.K, self.V = TextEnc(hp, self.L, training=training)  # (N, Tx, e)

                with tf.variable_scope("AudioEnc"):
                    self.Q = AudioEnc(hp, self.S, training=training)

                with tf.variable_scope("Attention"):
                    # R: (B, T/r, 2d)
                    # alignments: (B, N, T/r)
                    # max_attentions: (B,)
                    self.R, self.alignments, self.max_attentions = Attention(hp, self.Q, self.K, self.V,
                                                                             mononotic_attention=(not training),
                                                                             prev_max_attentions=self.prev_max_attentions)
                with tf.variable_scope("AudioDec"):
                    self.Y_logits, self.Y = AudioDec(hp, self.R, training=training) # (B, T/r, n_mels)
        else:  # num==2 & training. Note that during training,
            # the ground truth melspectrogram values are fed.
            with tf.variable_scope("SSRN"):
                self.Z_logits, self.Z = SSRN(hp, self.mels, training=training)

        if not training:
            # During inference, the predicted melspectrogram values are fed.
            with tf.variable_scope("SSRN"):
                self.Z_logits, self.Z = SSRN(hp, self.Y, training=training)

        with tf.variable_scope("gs"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if training:
            if num==1: # Text2Mel
                # mel L1 loss
                self.loss_mels = tf.reduce_mean(tf.abs(self.Y - self.mels))

                if hp.use_bd1_loss:
                    # mel binary divergence loss
                    self.loss_bd1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Y_logits, labels=self.mels))

                # guided_attention loss
                self.A = tf.pad(self.alignments, [(0, 0), (0, hp.max_N), (0, hp.max_T)], mode="CONSTANT", constant_values=-1.)[:, :hp.max_N, :hp.max_T]
                self.attention_masks = tf.to_float(tf.not_equal(self.A, -1))
                self.loss_att = tf.reduce_sum(tf.abs(self.A * self.gts) * self.attention_masks)
                self.mask_sum = tf.reduce_sum(self.attention_masks)
                self.loss_att /= self.mask_sum

                # total loss
                if hp.use_bd1_loss:
                    self.loss = self.loss_mels + self.loss_bd1 + self.loss_att
                else:
                    self.loss = self.loss_mels + self.loss_att

                tf.summary.scalar('train/loss_mels', self.loss_mels)
                if hp.use_bd1_loss: tf.summary.scalar('train/loss_bd1', self.loss_bd1)
                tf.summary.scalar('train/loss_att', self.loss_att)
                tf.summary.image('train/mel_gt', tf.expand_dims(tf.transpose(self.mels[:1], [0, 2, 1]), -1))
                tf.summary.image('train/mel_hat', tf.expand_dims(tf.transpose(self.Y[:1], [0, 2, 1]), -1))
            else: # SSRN
                # mag L1 loss
                self.loss_mags = tf.reduce_mean(tf.abs(self.Z - self.mags))

                if hp.use_bd2_loss:
                    # mag binary divergence loss
                    self.loss_bd2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Z_logits, labels=self.mags))

                # total loss
                if hp.use_bd2_loss:
                    self.loss = self.loss_mags + self.loss_bd2
                else:
                    self.loss = self.loss_mags
                    
                tf.summary.scalar('train/loss_mags', self.loss_mags)
                if hp.use_bd2_loss: tf.summary.scalar('train/loss_bd2', self.loss_bd2)
                tf.summary.image('train/mag_gt', tf.expand_dims(tf.transpose(self.mags[:1], [0, 2, 1]), -1))
                tf.summary.image('train/mag_hat', tf.expand_dims(tf.transpose(self.Z[:1], [0, 2, 1]), -1))

            # Training Scheme
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

#### for debugging:
from networks_min import TextEncMin1, TextEncMin2, TextEncMin3, TextEncMin16, TextEncMin26, TextEncMin36


class GraphTEOnly:
    def __init__(self, hp):
     
        self.L = tf.placeholder(tf.int32, shape=(None, None))

        with tf.variable_scope("Text2Mel"):
            with tf.variable_scope("TextEnc"):
                self.K, self.V = TextEnc(hp, self.L, training=False) 







class GraphTEOnly01:
    def __init__(self, hp):
     
        self.L = tf.placeholder(tf.int32, shape=(None, None))

        with tf.variable_scope("Text2Mel"):
            with tf.variable_scope("TextEnc"):
                self.O = TextEncMin1(hp, self.L, training=False) 




class GraphTEOnly02:
    def __init__(self, hp):
     
        self.L = tf.placeholder(tf.int32, shape=(None, None))

        with tf.variable_scope("Text2Mel"):
            with tf.variable_scope("TextEnc"):
                self.O = TextEncMin2(hp, self.L, training=False) 




class GraphTEOnly03:
    def __init__(self, hp):
     
        self.L = tf.placeholder(tf.int32, shape=(None, None))

        with tf.variable_scope("Text2Mel"):
            with tf.variable_scope("TextEnc"):
                self.O = TextEncMin3(hp, self.L, training=False) 



class GraphTEOnly16:
    def __init__(self, hp):
     
        self.L = tf.placeholder(tf.int32, shape=(None, None))

        with tf.variable_scope("Text2Mel"):
            with tf.variable_scope("TextEnc"):
                self.O = TextEncMin16(hp, self.L, training=False) 



class GraphTEOnly26:
    def __init__(self, hp):
     
        self.L = tf.placeholder(tf.int32, shape=(None, None))

        with tf.variable_scope("Text2Mel"):
            with tf.variable_scope("TextEnc"):
                self.O = TextEncMin26(hp, self.L, training=False) 



class GraphTEOnly36:
    def __init__(self, hp):
     
        self.L = tf.placeholder(tf.int32, shape=(None, None))

        with tf.variable_scope("Text2Mel"):
            with tf.variable_scope("TextEnc"):
                self.O = TextEncMin36(hp, self.L, training=False) 





          