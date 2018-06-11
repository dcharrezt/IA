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

// void NeuralNetwork::create( int numTrainingInputs, int numInputNeurons, 
// 						int numOutputs, int *hiddenLayers, int numHiddenLayers)
// {
// 	inputLayer.create(  )
// }

void NeuralNetwork::forwardPropagation( float *input )
{

}
float NeuralNetwork::backwardPropagation( float *targetOutput, float *inputs, 
													float learningRate )
{

}

int main(int argc, char const *argv[])
{
	
	return 0;
}
