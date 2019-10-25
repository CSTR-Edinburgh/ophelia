

'''
Install Flite for Ophelia like this:

cd ./ophelia/tool
git clone http://github.com/festvox/flite
cd flite
./configure
make
'''
import sys
import os
import re
import subprocess

from mcd import dtw  ## Matt Shannon's mcd is already a required part of the environment
                     ## so we might as well use it for aligning phones... 

HERE = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
flite_binary = os.path.realpath(os.path.abspath(HERE + '/../../tool/flite/bin/flite'))

if (not os.path.isfile(flite_binary)) or (not os.access(flite_binary, os.X_OK)):
    print '\n\n\n\nFlite binary does not exist at %s or is not executable'%(flite_binary)
    print 'See installation instructions at top of %s\n\n\n\n'%(__file__)
    sys.exit('!!!!!')


def simple_merge(phones, wordprons):
    print wordprons
    print phones
    #sys.exit('===ev=reb')
    output = []
    for chunk in phones:
        print '==='
        print chunk 
        if chunk=='pau':
            output.append(chunk)
        else:
            while len(chunk) > 0:
                print '----'
                print (chunk[:len(wordprons[0])], wordprons[0])
                if chunk[:len(wordprons[0])] == wordprons[0]:
                    output.append(chunk[:len(wordprons[0])])
                    chunk = chunk[len(wordprons[0]):]  # consume chnk
                    chunk = chunk.strip()
                    del wordprons[0]    ## consume pron
                else:
                    sys.exit('licnldnv')
    print [output]


def dtw_merge(phones, wordprons):
    # print phones
    # print wordprons
    # ['pau', 'ay w aa n t ax d t ax l iy d', 'pau']
    # ['ay', 'w aa n t ax d', 't ax', 'l iy d']

    wordprons = ' <> '.join(wordprons)
    wordprons = wordprons.split(' ')
    phones = ' '.join(phones)
    phones = phones.split(' ')

    print phones
    print wordprons

    def cost(x, y):
        if x==y:
            return 0
        elif x=='pau' and y=='<>':
            return 0
        else:
            return 1

    score, path = dtw.dtw(phones, wordprons, cost)
    print path
    output= []
    for (phone_i, iso_i) in path:
        print '%s %s'%(phones[phone_i], wordprons[iso_i])
        if wordprons[iso_i] == '<>':
            if phones[phone_i] == 'pau':
                output.append('pau')
            else:
                output.append('<>')
        else:
            output.append(phones[phone_i])
    print output


def get_flite_phonetisation(textfile, dictionary='cmulex'):
    #return '<_START_> dh ax <> b er ch <_END_>'


    os.system("sed 's/ /, /g' %s > %s.tmp"%(textfile, textfile))
    comm = "%s -f %s.tmp -ps none"%(flite_binary, textfile)
    print comm
    wordphones = subprocess.check_output(comm, shell=True)
    wordprons = re.split('\s*pau\s*', wordphones)
    wordprons = [w.strip() for w in wordprons if w]

    # print wordprons  # ['dh ax', 'f ih sh', 't w ih s t ax d', 'ae n d', 't er n d']

    comm = "%s -f %s -ps none"%(flite_binary, textfile)
    print comm
    phones = subprocess.check_output(comm, shell=True)
    phones = re.split('(\s*pau\s*)', phones)
    phones = [w.strip() for w in phones if w]


    phones_no_pau = [w for w in phones if w != 'pau']
    if 0: # ' '.join(phones_no_pau) == ' '.join(wordprons):
        print 'Pron of words in context matches pronunciation of isolated words - proceed with simple merge'
        simple_merge(phones, wordprons)
    else:
        print 'Back off to dynamic programming  ;-)'
        dtw_merge(phones, wordprons)



if 1: ## dev and debug
    os.system('echo "The fish; twisted and turned." > /tmp/test.txt')
    #os.system('echo "I wanted to lead." > /tmp/test.txt')
    get_flite_phonetisation('/tmp/test.txt', dictionary='cmulex')
    sys.exit('evcrv')