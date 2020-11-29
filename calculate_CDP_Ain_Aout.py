
import sys
import math
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def load_binary_file(file_name, dimension):

	fid_lab = open(file_name, 'rb')
	features = np.fromfile(fid_lab, dtype=np.float32)
	fid_lab.close()
	assert features.size % float(dimension) == 0.0,'specified dimension not compatible with data'
	frame_number = int(features.size / dimension)
	features = features[:(dimension * frame_number)]
	features = features.reshape((-1, dimension))

	features = np.transpose(features)
	return  features, frame_number

def get_att_per_input(A):

	att_per_input = np.trim_zeros(np.sum(A,axis=1),'b')
	num_input = len(att_per_input)

	return att_per_input, num_input

# coverage deviation penalty: punishes when input token is not represented by output or overly represented
# (lack of attention) skips and (too much attention) prolongation
def getCDP(A):

	att_per_input, num_input = get_att_per_input(A)
	tmp = (1. - att_per_input )**2
	tmp = np.sum( np.log( 1. + (1. - att_per_input )**2 ) )
	CDP = tmp / num_input # removed the minus sign

	return CDP

def getEnt(A):

	Entr = 0.0

	for a in A: # traverse rows of A
		norm   = np.sum(a)
		normPd = a
		if norm != 0.0:
			normPd = [ p / norm for p in a ]
		entr = np.sum( [ ( p * np.log(p) if (p!=0.0) else 0.0 ) for p in normPd ] )
		Entr += entr

	Entr = -Entr/A.shape[0]
	Entr /= np.log(A.shape[1]) # to normalise it between 0-1

	return Entr


# absentmindess penalty: punishes scattered attention, dispersion is calculated via the entropy
def getAP(A):

	att_per_input, num_input = get_att_per_input(A)
	num_output = A.shape[1]
	A     = A[:num_input,:]
	APout = 0.0
	APin  = 0.0

	APin  = getEnt(A)
	APout = getEnt(np.transpose(A))
	
	return APin, APout

def main(file_name):
		
	dim  = int(file_name.split('_')[-1].split('.')[0])

	A, fn = load_binary_file(file_name, dim) # square matrix axis 0: input / axis 1: output

	# fig, ax = plt.subplots()
	# im = ax.imshow(A)
	# fig.colorbar(im)
	# plt.ylabel('Encoder timestep')
	# plt.xlabel('Decoder timestep')
	# plt.show()

	CDP = getCDP(A)
	APin, APout = getAP(A)
	
	print('CDP: ' + str(CDP)) 
	print('Ain: ' + str(APin))
	print('Aout: ' + str(APout))

if __name__ == '__main__':

	# Input attention file - float format

	file_name = sys.argv[1]
	
	main(file_name)

