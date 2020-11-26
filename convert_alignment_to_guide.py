
import sys
import math
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from libutil import save_floats_as_8bit
import tqdm
from concurrent.futures import ProcessPoolExecutor

gD = 0.2
gW = 0.1

DEBUG = False

def main(file_name,out_file):

	F = np.load(file_name)
	F = np.transpose(F)

	ndim, tdim = F.shape # x: encoder (N) / y: decoder (T)

	## Convert alignment to attention guide
	if DEBUG:
		D = np.zeros((ndim, tdim), dtype=np.float32) # diagonal guide
	W = np.zeros((ndim, tdim), dtype=np.float32) # alignment based guide

	for n_pos in range(ndim):
		for t_pos in range(tdim):
			
			n_pos_new = np.argmax(F[:,t_pos])
			W[n_pos,t_pos] = 1 - np.exp( -(n_pos / float(ndim) - n_pos_new / float(ndim) ) ** 2  / (2 * gW * gW))

			if DEBUG:
				D[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(tdim) - n_pos / float(ndim)) ** 2 / (2 * gD * gD))

	## Smooth the alignment based guide
	S = gaussian_filter(W, sigma=1) # trying to blur
	# needs min max norm here to make sure 0-1
	S = ( S - S.min()) / ( S.max() - S.min() )

	save_floats_as_8bit(S, out_file)

	if DEBUG:

		D = ( D - D.min()) / ( D.max() - D.min() )
		W = ( W - W.min()) / ( W.max() - W.min() )

		for plot_type in range(0,3):

			## Visualization 
			if plot_type==0:
				M = F+D # add forced alignment path to help visualisation
			elif plot_type == 1:
				M = F+W # add forced alignment path to help visualisation
			elif plot_type == 2:
				M = F+S # add forced alignment path to help visualisation

			fig, ax = plt.subplots()
			im = ax.imshow(M)
			# plt.title('Diagonal (top), Alignment based (middle), Alignment based smoothed (bottom)', fontsize=8)
			fig.colorbar(im,fraction=0.035, pad=0.04)
			plt.ylabel('Encoder timestep', fontsize=12)
			plt.xlabel('Decoder timestep', fontsize=12)

			if plot_type==0:
				plt.title('Diagonal attention guide', fontsize=12)
				plt.savefig('attention_guide_diagonal.pdf')
			elif plot_type == 1:
				plt.title('Forced alignment based attention guide', fontsize=12)
				plt.savefig('attention_guide_fa.pdf')
			elif plot_type == 2:
				plt.title('Forced alignment based attention guide (smoothed)', fontsize=12)
				plt.savefig('attention_guide_fa_smooth.pdf')

			plt.show()

if __name__ == '__main__':

	# Usage: python convert_alignment_to_guide.py CB-EM-55-07.npy CB-EM-55-07_out.npy 

	inputdir   = sys.argv[1]
	outputdir  = sys.argv[2]
	ncores = 5
	executor = ProcessPoolExecutor(max_workers=ncores)    
	futures  = []
	for file in glob.iglob(inputdir + '/*.npy'):
		outfile = outputdir + file.split('/')[-1]
		futures.append(executor.submit(main, file, outfile))

	proc_list = [future.result() for future in tqdm.tqdm(futures)]

