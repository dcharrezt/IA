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

void NeuralNetwork::create( int numLayerInputs, int numInputLayerNeurons, 
			int numOutputLayerNeurons, int *sizesHiddenLayers, int numHiddenLayers)
{
	inputLayer.create( numLayerInputs, numInputLayerNeurons );
	if( hiddenLayers && numHiddenLayers)
	{
		this->hiddenLayers = new Layer*[ numHiddenLayers ];
		this->numHiddenLayers = numHiddenLayers;
		for (int i = 0; i < numHiddenLayers; ++i)
		{
			hiddenLayers[i] = new Layer;
			if( i == 0 )
				this->hiddenLayers[i]->create( numInputLayerNeurons, sizesHiddenLayers[i] );
			else
				this->hiddenLayers[i]->create( sizesHiddenLayers[i-1], sizesHiddenLayers[i] );
		}
		outputLayer.create( sizesHiddenLayers[numHiddenLayers-1], numOutputLayerNeurons );
	}
	else
		outputLayer.create( numInputLayerNeurons, numOutputLayerNeurons );
}

void NeuralNetwork::forwardPropagation( float *input )
{
	memcpy( inputLayer.layerInputs, input, inputLayer.numLayerInputs*sizeof(float));

}

float NeuralNetwork::backwardPropagation( float *targetOutput, float *inputs, 
													float learningRate )
{

}

void NeuralNetwork::updateNextLayerInput( int layerIndex )
{
	if( layerIndex == -1 )
	{
		for (int i = 0; i < inputLayer.numNeurons ; ++i)
		{
			if( hiddenLayers )
				hiddenLayers[0]->layerInputs[i] = inputLayer.neurons[i]->neuronOut;
			else
				outputLayer.layerInputs[i] = inputLayer.neurons[i]->neuronOut;
		}
	}
	else
	{
		for (int i = 0; i < hiddenLayers[layerIndex]->numNeurons; ++i)
		{
			if( layerIndex < numHiddenLayers -1 )
			{
				hiddenLayers[layerIndex+1]->layerInputs[i] = 
							hiddenLayers[layerIndex]->neurons[i]->neuronOut;
			}
			else
				outputLayer.layerInputs[i] = 
							hiddenLayers[layerIndex]->neurons[i]->neuronOut;
		}
	}
}

int main(int argc, char const *argv[])
{
	
	return 0;
}