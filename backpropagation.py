import math
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
# from sklearn import preproccesing


n_inputs = 4
n_hidden_layers = [2, 4]
n_outputs = 3


def sigmoid( x ):
	return 1 / ( 1 + math.exp(-x) )

def sigmoid_derivative( x ):
	return math.exp( x ) / (1+math.exp(-x))**2

def initialize_weights(n_inputs, n_hidden_layers, n_outputs):
	current_input = n_inputs
	network = []
	# hidden_layers = [ [ [ random.random() for i in range(k) ] \
	# 							for j in range(k+1)] for k in n_hidden_layers]
	hidden_layers = []
	for i in n_hidden_layers:
		for j in range( i ):
			hidden_layers.append( [ random.random() for k in range(current_input+1) ] )
		current_input = i

	# output_layer = [ random.random() for i in range(n_outputs) ] 
	print( hidden_layers )

# def forward_propagation():




if __name__ == "__main__":

	iris = datasets.load_iris()

	X_train, X_test, y_train, y_test = train_test_split(
				iris.data, iris.target, test_size=0.4, random_state=0)

	initialize_weights( n_inputs, n_hidden_layers, n_outputs)

	# print(y_train)
	# print(len(y_train))
	# print("asdasdmflgsflgsdfijsndmfkoshdjfknglsdffl")
	# print(y_test)

	# print(X_train)
	# print(len(X_train))

	# print(y_train)
	# print(len(y_train))