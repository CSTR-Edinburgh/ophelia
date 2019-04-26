


import fileinput, sys

infile = sys.argv[1]

i = 0
for line in fileinput.input(infile):
    name, t1, t2, phones, speaker, durs = line.strip(' \n').split('|') # [-1]
    durs = durs.split(' ')
    assert durs[0] == '24'
    assert durs[-2] == '24'
    assert durs[-1] == '0'
    durs[0] = '0'
    durs[-2] = '0'
    durs[-1] = '0'
    # if i==5:
    #    break
    durs = ' '.join(durs)
    line2 = '|'.join([name, t1, t2, phones, speaker, durs])
    print line2
    i += 1
# print i
