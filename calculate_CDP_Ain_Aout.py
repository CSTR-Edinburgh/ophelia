
import sys
import math
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

	A = np.load(file_name) # matrix axis 0: input (phones) / axis 1: output (acoustic)
        print(A.shape)
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

	# Input attention matrix - numpy format - axis 0: input (phones) / axis 1: output (acoustic)

	file_name = sys.argv[1]
	
	main(file_name)


