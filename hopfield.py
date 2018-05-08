import numpy as np

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