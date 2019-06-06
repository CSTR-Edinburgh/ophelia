#!/usr/bin/env python

from argparse import ArgumentParser


def main_work():

    a = ArgumentParser()
    a.add_argument('-i', dest='infile', required=True)
    a.add_argument('-o', dest='outfile', required=True)
    opts = a.parse_args()

    o = open(opts.outfile, 'w')
    
    with open(opts.infile, 'r') as f:
        for line in f.readlines()[:-1]:
            if line[3] == '_':  # if clauses dealing with different length p-numbers (not really applicable for public VCTK)
                speaker_id = line[0:3]
            elif line[4] == '_':
                speaker_id = line[0:4]
            elif line[5] == '_':
                speaker_id = line[0:5]
            else:
                print('Something is wrong with the input file - speaker ID cannot be parsed!')
            o.write('{}|{}\n'.format(line.rstrip(), speaker_id))

    o.close()

if __name__ == "__main__":
    main_work()
    
