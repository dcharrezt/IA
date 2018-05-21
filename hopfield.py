import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_mldata

def get_mnist_train_data(n_patterns):
    mnist = fetch_mldata('MNIST original')
    mnist.data = mnist.data.astype(np.float32)
    mnist.data /= 255
    index_list = [ list(mnist.target).index(i) for i in range(n_patterns) ]
    train = [mnist.data[i] for i in index_list]
    return train

def get_mnist_test_data(train, error_rate):
    test = np.copy(train)
    for i, t in enumerate(test):
        s = np.random.binomial(1, error_rate, len(t))
        for j in range(len(t)):
            if s[j] != 0:
                t[j] *= 0
    return test

def create_weight_matrix( x ):
	matrix_length = len( x[0] )
	matrix = np.zeros((matrix_length, matrix_length))
	
	for i in range(0, matrix_length):
		for j in range(0, matrix_length):
			if i != j:
				s = 0
				for k in x:
					s += k[i]*k[j]
				matrix[i][j] = s
	return matrix

def activation_function( x ):
	if( x >= 0 ):
		return 1
	else:
		return -1

def activation( m ):
	matrix_length = len( m[0] )
	for i in range(0, matrix_length):
		for j in range(0, matrix_length):
			if i != j:
				matrix[i][j] = activation_function(matrix[i][j])

def get_output( x, matrix ):
	tmp = np.dot([x], matrix)
	tmp = [ activation_function(i) for i in tmp[0] ]
	print( "input evualued: \t", tmp )
	return tmp

if __name__ == "__main__":
	inp = [[1,1,1], [-1,-1,-1]]
	imp_t = [-1, 1, -1]

	matrix = create_weight_matrix( inp )
	activation( matrix )

	print( matrix )

	get_output( imp_t, matrix )


    n_patterns = 10
    n_units = 28
    error_rate = 0.1

    train = mnist4hn(n_patterns)
    fig, ax = plt.subplots(1, n_patterns, figsize=(5,2))
    for i in range(n_patterns):
        ax[i].matshow(train[i].reshape((n_units, n_units)), cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()

    test = addnoise(train, error_rate)
    fig, ax = plt.subplots(1, n_patterns, figsize=(5,2))
    for i in range(n_patterns):
        ax[i].matshow(test[i].reshape((n_units, n_units)), cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()
