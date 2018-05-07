import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preproccesing


def sigmoid( x ):
	return 1 / ( 1 + math.exp(-x) )

def sigmoid_derivative( x ):
	return math.exp( x ) / (1+math.exp(-x))**2



if __name__ == "__main__":

	iris = datasets.load_iris()

	X_train, X_test, y_train, y_test = train_test_split(
				iris.data, iris.target, test_size=0.4, random_state=0)

	print(y_train)
	print(len(y_train))
	print("asdasdmflgsflgsdfijsndmfkoshdjfknglsdffl")
	print(y_test)

	print(X_train)
	print(len(X_train))