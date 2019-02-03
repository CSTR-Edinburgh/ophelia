import logging
import os
import sys
import subprocess
import socket
import numpy
import tensorflow
from libutil import safe_makedir

def logger_setup(logdir):

    safe_makedir(logdir)

    ## Get new unique named logfile for each run:
    i = 1
    while True:
        logfile = os.path.join(logdir, 'log_{:06d}.txt'.format(i))
        if not os.path.isfile(logfile):
            break
        else:
            i += 1

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s | %(threadName)-3.3s | %(levelname)-1.1s | %(message)s')

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info('Set up logger to write to console and %s'%(logfile))

    log_environment_information(logger, logfile)


def log_environment_information(logger, logfile):
    ### This function's contents adjusted from Merlin (https://github.com/CSTR-Edinburgh/merlin/blob/master/src/run_merlin.py)
    ### TODO: other things to log here?
    logger.info('Installation information:')
    logger.info('  Merlin directory: '+os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
    logger.info('  PATH:')
    env_PATHs = os.getenv('PATH')
    if env_PATHs:
        env_PATHs = env_PATHs.split(':')
        for p in env_PATHs:
            if len(p)>0: logger.info('      '+p)
    logger.info('  LD_LIBRARY_PATH:')
    env_LD_LIBRARY_PATHs = os.getenv('LD_LIBRARY_PATH')
    if env_LD_LIBRARY_PATHs:
        env_LD_LIBRARY_PATHs = env_LD_LIBRARY_PATHs.split(':')
        for p in env_LD_LIBRARY_PATHs:
            if len(p)>0: logger.info('      '+p)
    logger.info('  Python version: '+sys.version.replace('\n',''))
    logger.info('    PYTHONPATH:')
    env_PYTHONPATHs = os.getenv('PYTHONPATH')
    if env_PYTHONPATHs:
        env_PYTHONPATHs = env_PYTHONPATHs.split(':')
        for p in env_PYTHONPATHs:
            if len(p)>0:
                logger.info('      '+p)
    logger.info('  Numpy version: '+numpy.version.version)
    logger.info('  Tensorflow version: '+tensorflow.__version__)
    #logger.info('    THEANO_FLAGS: '+os.getenv('THEANO_FLAGS'))
    #logger.info('    device: '+theano.config.device)

    # Check for the presence of git
    ret = os.system('git status > /dev/null')
    if ret==0:
        logger.info('  Git is available in the working directory:')
        git_describe = subprocess.Popen(['git', 'describe', '--tags', '--always'], stdout=subprocess.PIPE).communicate()[0][:-1]
        logger.info('    DC_TTS_OSW version: {}'.format(git_describe))
        git_branch = subprocess.Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=subprocess.PIPE).communicate()[0][:-1]
        logger.info('    branch: {}'.format(git_branch))
        git_diff = subprocess.Popen(['git', 'diff', '--name-status'], stdout=subprocess.PIPE).communicate()[0]
        if sys.version_info.major >= 3:
            git_diff = git_diff.decode('utf-8')
        git_diff = git_diff.replace('\t',' ').split('\n')
        logger.info('    diff to DC_TTS_OSW version:')
        for filediff in git_diff:
            if len(filediff)>0: logger.info('      '+filediff)
        logger.info('      (all diffs logged in '+os.path.basename(logfile)+'.gitdiff'+')')
        os.system('git diff > '+logfile+'.gitdiff')

    logger.info('Execution information:')
    logger.info('  HOSTNAME: '+socket.getfqdn())
    logger.info('  USER: '+os.getenv('USER'))
    logger.info('  PID: '+str(os.getpid()))
    PBS_JOBID = os.getenv('PBS_JOBID')
    if PBS_JOBID:
        logger.info('  PBS_JOBID: '+PBS_JOBID)

