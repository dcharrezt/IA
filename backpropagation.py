import math
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
# from sklearn import preproccesing

n_inputs = 4
n_hidden_layers = [4, 4]
n_outputs = 3
n_epochs = 500
learning_rate = 0.5

def sigmoid( x ):
	return 1. / ( 1. + math.exp(-x) )

def sigmoid_derivative( x ):
	return x * ( 1. - x )

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
	return current_input

def square_error( target_output, output ):
	error = 0.
	for i in range( len(output) ):
		error += 0.5 * ( ( target_output[i] - output[i] ) ** 2 )
	return error

def backward_propagation( network, y_train ):
	for i in reversed( range(len(network)) ):
		errors = []
		if i != len(network)-1 :
			for j in range( len(network[i]) ):
				err = 0.
				for neuron in network[i+1]:
					err += ( neuron['weights'][j] * neuron['delta'] )
				errors.append( err )
		else:
			for j in range( len(network[i]) ):
				errors.append(  y_train[j] - network[i][j]['output'] )
		for j in range( len(network[i]) ):
			network[i][j]['delta'] = errors[j] * \
							sigmoid_derivative( network[i][j]['output'])

def update_weights( network, inputs ):
	for i in range( len(network) ):
		if i != 0:
			inputs = [ neuron['output'] for neuron in network[i-1] ]
		for neuron in network[i]:
			for j in range( len( inputs) ):
				neuron['weights'][j] += learning_rate*neuron['delta']*inputs[j]
			neuron['weights'][-1] += learning_rate*neuron['delta']

def train_network( network, X_train, y_train, n_epochs ):
	for i in range( n_epochs ):
		error = 0.
		for inputs, out in zip(X_train, y_train):
			outputs = forward_propagation( inputs, network )
			target = [0] *  n_outputs 
			target[ out ] = 1
			error += square_error( target, outputs[:-1] )
			backward_propagation( network, target )
			update_weights( network, inputs )
		print("Epoch: "+str(i)+"\tError: "+str(error))

if __name__ == "__main__":

	iris = datasets.load_iris()

	X_train, X_test, y_train, y_test = train_test_split(
				iris.data, iris.target, test_size=0.4, random_state=0)

	# print(y_train)

	network = initialize_weights( n_inputs, n_hidden_layers, n_outputs)
	# outputs = forward_propagation( X_train, network )
	# backward_propagation( network, [0,0,1] )
	train_network( network, X_train, y_train, n_epochs )