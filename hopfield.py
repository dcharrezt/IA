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

def activation( m ):
	matrix_length = len( m[0] )
	for i in range(0, matrix_length):
		for j in range(0, matrix_length):
			if i != j:
				if( matrix[i][j] >= 0 ):
					matrix[i][j] = 1
				else:
					matrix[i][j] = -1
	print( m )

def get_output( x ):
	print( np.dot([x], matrix) )

if __name__ == "__main__":
	inp = [[1,1,1], [-1,-1,-1]]
	imp_t = [-1,1-1]

	matrix = create_weight_matrix( inp )
	activation( matrix )
	get_output( imp_t )