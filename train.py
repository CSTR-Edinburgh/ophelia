# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch, load_vocab
from modules import *
from networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN
import tensorflow as tf
from utils import *
import sys


## to import configs

HERE = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
sys.path.append( HERE + '/config/' )

import importlib

from argparse import ArgumentParser

from graphs import Graph



def main_work():

    #################################################
            
    # ============= Process command line ============


    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-m', dest='num', required=True, type=int, choices=[1, 2], \
                    help='1: Text2mel, 2: SSRN')
    opts = a.parse_args()
    
    # ===============================================
    num = opts.num
    config = opts.config

    conf_mod = importlib.import_module(config)
    hp = conf_mod.Hyperparams()

    g = Graph(hp, num=num); print("Training Graph loaded")

    logdir = hp.logdir + "-" + str(num)
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step)

    save_every = hp.save_every_n_iterations
    with sv.managed_session() as sess:
        
        if hp.restart_from_savepath:
            sess.graph._unsafe_unfinalize() ## !!! https://stackoverflow.com/questions/41798311/tensorflow-graph-is-finalized-and-cannot-be-modified/41798401
            sess.run(tf.global_variables_initializer())

            print ('Restore parameters')
            if num==1:
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
                saver1 = tf.train.Saver(var_list=var_list)
                savepath = hp.restart_from_savepath + "-1"
                print(savepath)
                saver1.restore(sess, tf.train.latest_checkpoint(savepath))
                print("Text2Mel Restored!")
            elif num==2:
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                           tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
                saver2 = tf.train.Saver(var_list=var_list)
                savepath = hp.restart_from_savepath + "-2"
                saver2.restore(sess, tf.train.latest_checkpoint(savepath))
                print("SSRN Restored!")


        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

                # Write checkpoint files at every 1k steps

                if 0: print ('gs: %s'%(gs))
                if gs % save_every == 0:
                    sv.saver.save(sess, logdir + '/model_gs_{}'.format(str(gs // save_every).zfill(3) + "k"))

                    if num==1:
                        # plot alignment
                        alignments = sess.run(g.alignments)
                        plot_alignment(hp, alignments[0], str(gs // save_every).zfill(3) + "k", logdir)

                # break
                if gs > hp.num_iterations: break

    print("Done")


if __name__=="__main__":

    main_work()

