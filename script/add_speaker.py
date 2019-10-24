#!/usr/bin/env python

from argparse import ArgumentParser


def main_work():

    a = ArgumentParser()
    a.add_argument('-i', dest='infile', required=True)
    a.add_argument('-o', dest='outfile', required=True)
    opts = a.parse_args()

    outf = open(opts.outfile, 'w')
    transcript = open(opts.infile, 'r').read().split('\n')

    for line in transcript:
        spk = line.split('_')[0]
        outf.writelines(line+'||'+spk+'\n')
    outf.close()

if __name__ == "__main__":
    main_work()
