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
		layer = []
		for j in range( neurons_per_layer[i+1] ):
			layer.append({'weights':[ random.random() for i \
								in range ( neurons_per_layer[i] + 1 ) ]})
		network.append( layer )
	return network

def forward_propagation( X_train , network ):
	current_input = list(X_train) + [1]
	for layer in network:
		for neuron in layer:
			neuron['output'] = sigmoid(np.dot( current_input, neuron['weights']))
		current_input = [i['output'] for i in layer] + [1]

def square_error( target_output, output ):
	error = 0.
	for i in range( len(output) ):
		error += 0.5 * ( ( target_output[i] - output[i] ) ** 2 )
	return error

def backward_propagation( network, y_train ):
	for layer, next_layer in zip(reversed(network[:-1]), reversed(network)):
		errors = []
		if next_layer == network[-1]:
			for neuron, target in zip(layer, y_train):
				errors.append(  neuron['output'] - target )
		else:
			for i in range( len(layer) ):
				err = 0.
				for next_neuron in next_layer:
					err += ( next_neuron['weights'][i] * next_neuron['delta'] )
				errors.append( err )
		for neuron, error in zip(layer, errors):
			neuron['delta'] = error * sigmoid_derivative(neuron['output'])

if __name__ == "__main__":

	iris = datasets.load_iris()

	X_train, X_test, y_train, y_test = train_test_split(
				iris.data, iris.target, test_size=0.4, random_state=0)

	network = initialize_weights( n_inputs, n_hidden_layers, n_outputs)
	print(network)
	forward_propagation( [1,1], network )
	print( network )
	backward_propagation( network, [0, 1] )
	print( network )
	# print(y_train)
	# print(len(y_train))
	# print("asdasdmflgsflgsdfijsndmfkoshdjfknglsdffl")
	# print(y_test)

	# print(X_train)
	# print(len(X_train))

	# print(y_train)
	# print(len(y_train))