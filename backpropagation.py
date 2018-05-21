import math
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
# from sklearn import preproccesing

n_inputs = 2
n_hidden_layers = [3, 2]
n_outputs = 2


def sigmoid( x ):
	return 1 / ( 1 + math.exp(-x) )

def sigmoid_derivative( x ):
	return x*(1-x)

def initialize_weights(n_inputs, n_hidden_layers, n_outputs):
	network = []
	neurons_per_layer = list( n_hidden_layers )
	neurons_per_layer = [n_inputs] + neurons_per_layer + [n_outputs]
		
	for i in range( len(neurons_per_layer) - 1 ):
		network.append( np.random.rand( neurons_per_layer[i+1], \
					neurons_per_layer[i] + 1 ) ) 

	return network

def forward_propagation( X_train , network ):

	X_train_wbias = list(X_train) + [1]
	tmp_out = [] 
	for layer in network:
		tmp_out = [] 
		print("layer\t", layer)
		for neuron in layer:
			print("neuron\t", neuron)
			print("X\t", X_train_wbias)
			tmp_out.append( np.dot( X_train_wbias, neuron ) )
		tmp_out = [ sigmoid(i) for i in tmp_out]
		X_train_wbias = list(tmp_out) + [1]

	print("TMP", tmp_out)

def square_error( target_output, output ):
	error = 0.
	for i in range( len(output) ):
		error += 0.5 * ( ( target_output[i] - output[i] ) ** 2 )
	return error

if __name__ == "__main__":

	iris = datasets.load_iris()

	X_train, X_test, y_train, y_test = train_test_split(
				iris.data, iris.target, test_size=0.4, random_state=0)

	network = initialize_weights( n_inputs, n_hidden_layers, n_outputs)
	print(network)
	forward_propagation( [1,1], network )
	# print(y_train)
	# print(len(y_train))
	# print("asdasdmflgsflgsdfijsndmfkoshdjfknglsdffl")
	# print(y_test)

	# print(X_train)
	# print(len(X_train))

	# print(y_train)
	# print(len(y_train))