#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork(): hiddenLayers(0), numHiddenLayers(0)
{

}

NeuralNetwork::~NeuralNetwork()
{
	if( hiddenLayers )
	{
		for (int i = 0; i < numHiddenLayers; ++i)
		{
			delete hiddenLayers[i];
		}
		delete [] hiddenLayers;
	}
}

int main(int argc, char const *argv[])
{
	
	return 0;
}