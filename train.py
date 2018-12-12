# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from tqdm import tqdm


import glob
import shutil
from data_load import get_batch, load_vocab, load_data
from modules import *
from networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN
import tensorflow as tf
from utils import *
import sys


## to import configs
# HERE = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
# sys.path.append( HERE + '/config/' )
# import importlib
import imp


from argparse import ArgumentParser

from graphs import Graph



def synth_valid_batch(hp, sess, g, outdir, num=1):
    L = load_data(hp, mode="validation")

    if num==1:
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)

        for j in tqdm(range(hp.max_T)):
            _Y, _max_attentions, _alignments, = \
                sess.run([ g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions}) ## osw: removed global_step from synth loop
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]
    else:
        print ('pass!')
        pass


    #print ('get mag...')
    # Get magnitude
    #Z = sess.run(g.Z, {g.Y: Y})    


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
    config = os.path.abspath(opts.config)
    assert os.path.isfile(config)

    conf_mod = imp.load_source('config', config)
    hp = conf_mod.Hyperparams()


    g = Graph(hp, num=num); print("Training Graph loaded")

    logdir = hp.logdir + "-" + str(num)
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step)

    save_every = hp.save_every_n_iterations
    archive_every = hp.archive_every_n_iterations
    if archive_every:
        assert archive_every % save_every == 0
        if not os.path.isdir(logdir + '/archive/'):
            os.makedirs(logdir + '/archive/')
    with sv.managed_session() as sess:
        
        if hp.restart_from_savepath:
            print(hp.restart_from_savepath)
            
            assert len(hp.restart_from_savepath) == 2
            restart_from_savepath1, restart_from_savepath2 = hp.restart_from_savepath
            restart_from_savepath1 = os.path.abspath(restart_from_savepath1)
            restart_from_savepath2 = os.path.abspath(restart_from_savepath2)

            sess.graph._unsafe_unfinalize() ## !!! https://stackoverflow.com/questions/41798311/tensorflow-graph-is-finalized-and-cannot-be-modified/41798401
            sess.run(tf.global_variables_initializer())

            print ('Restore parameters')
            if num==1:
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
                saver1 = tf.train.Saver(var_list=var_list)
                #savepath = hp.restart_from_savepath + "-1"
                #print(savepath)
                saver1.restore(sess, restart_from_savepath1)
                print("Text2Mel Restored!")
            elif num==2:
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                           tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
                saver2 = tf.train.Saver(var_list=var_list)
                #savepath = hp.restart_from_savepath + "-2"
                saver2.restore(sess, restart_from_savepath2)
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

                if archive_every > 0:
                    if gs % archive_every == 0:
                        stem = logdir + '/model_gs_{}'.format(str(gs // save_every).zfill(3) + "k")
                        for fname in glob.glob(stem + '*'):
                            shutil.copy(fname, logdir + '/archive/')
                        #synth_valid_batch(hp, sess, g, logdir + '/archive/', num=num)

                    
                # break
                #print (gs, hp.num_iterations)
                if gs > hp.num_iterations: return

    print("Done")


if __name__=="__main__":

    main_work()

