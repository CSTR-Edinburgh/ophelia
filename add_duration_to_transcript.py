
import numpy as np
import os
import sys
import os.path

# Usage: python add_duration_to_transcript.py fa_matrix_dir transcript_file new_transcript_file
fa_matrices_dir = sys.argv[1]
transcript = sys.argv[2]
out_transcript = sys.argv[3]

with open(transcript, 'r') as f:
	tra = f.readlines()

f = open(out_transcript,'w')


for t in tra:

	file = t.split('|')[0]
	phones = t.split('|')[3]
	num_phones = len(phones.split(' '))
        
        if os.path.exists(fa_matrices_dir + file + '.npy'):

  	   A = np.load( fa_matrices_dir + file + '.npy')
	   durations = np.sum(A,axis=0)
	   durations = 4*durations # durations are given for 12.5ms frames, convert to 50ms
	   assert(num_phones == A.shape[1])
	   assert(num_phones == len(durations))

	   f.write(t.rstrip() + '||' + " ".join(map(str, durations.astype(int))) + '\n')

        else:
           print("Missing " + file)

f.close()
