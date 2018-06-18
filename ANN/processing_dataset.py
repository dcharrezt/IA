import pandas as pd
import math
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets


if __name__ == "__main__":

	iris = datasets.load_iris()
	scaler = MinMaxScaler(copy=True, feature_range=(0, 1))

	X_train, X_test, y_train, y_test = train_test_split(
				iris.data, iris.target, test_size=0.4, random_state=123)
				
	print( "X TRAIN ", X_train )
	print( "X TRAIN NORMALIZER ", scaler.fit( X_train ) )
	print(scaler.data_max_)
	print(scaler.data_min_)
	print( "X TRAIN NORMALIZER ", scaler.transform( X_train ) )
	normalized_X_train = scaler.transform( X_train )
	print( "X TEST ", X_test)
	print( "X TEST NORMALIZER ", scaler.fit( X_test ) )
	print(scaler.data_max_)
	print(scaler.data_min_)
	print( "X TEST NORMALIZER ", scaler.transform( X_test ) )
	normalized_X_test = scaler.transform( X_test )
	
	with open('dataset/iris_x_train_60p.csv', 'wb') as FOUT:
		np.savetxt(FOUT, normalized_X_train, delimiter=',', fmt='%f')
	
	with open('dataset/iris_x_test_40p.csv', 'wb') as FOUT:
		np.savetxt(FOUT, normalized_X_test, delimiter=',', fmt='%f')
	
#	print( "Y TRAIN ", y_train)
#	print( "Y TEST ", y_test)
	
	y_test_to_file = []
	for i in y_test:
		target = [0] *  3
		target[ i ] = 1
		y_test_to_file.append( target )
	
#	print( "Target ", y_test_to_file )
	
	with open('dataset/iris_y_test_40p.csv', 'wb') as FOUT:
		np.savetxt(FOUT, y_test_to_file, delimiter=',', fmt='%f')
	
	y_train_to_file = []
	for i in y_train:
		target = [0] *  3
		target[ i ] = 1
		y_train_to_file.append( target )
	
#	print( "Target ", y_train_to_file )
	
	with open('dataset/iris_y_train_60p.csv', 'wb') as FOUT:
		np.savetxt(FOUT, y_train_to_file, delimiter=',', fmt='%f')
		
	
		
		
		
		
		
		
		
		
		

