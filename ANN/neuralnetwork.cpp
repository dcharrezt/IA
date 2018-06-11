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
			int numOutputLayerNeurons, int *sizesHiddenLayers, int numHiddenLayers )
{
	// activationFunctions[0] = NeuralNetwork::sigmoid;
	// activationFunctions[1] = NeuralNetwork::gaussian;
	float (*activationFunctions[2])(float) = {sigmoid, gaussian};

	inputLayer.create( numLayerInputs, numInputLayerNeurons, activationFunctions[0]);
	if( hiddenLayers && numHiddenLayers)
	{
		this->hiddenLayers = new Layer*[ numHiddenLayers ];
		this->numHiddenLayers = numHiddenLayers;
		for (int i = 0; i < numHiddenLayers; ++i)
		{
			hiddenLayers[i] = new Layer;
			if( i == 0 )
			{
				this->hiddenLayers[i]->create( numInputLayerNeurons,
						 sizesHiddenLayers[i], activationFunctions[0] );
			}
			else
			{
				this->hiddenLayers[i]->create( sizesHiddenLayers[i-1], 
					sizesHiddenLayers[i], activationFunctions[0] );
			}
		}
		outputLayer.create( sizesHiddenLayers[numHiddenLayers-1], 
							numOutputLayerNeurons , activationFunctions[0]);
	}
	else
		outputLayer.create( numInputLayerNeurons, numOutputLayerNeurons, 
										activationFunctions[0] );
}

void NeuralNetwork::forwardPropagation( float *input )
{
	memcpy( inputLayer.layerInputs, input, inputLayer.numLayerInputs*sizeof(float));
	inputLayer.getActivation();

	updateNextLayerInput(-1);
	if( hiddenLayers )
	{
		for (int i = 0; i < numHiddenLayers; ++i)
		{
			hiddenLayers[i]->getActivation();
			updateNextLayerInput(i);
		}
	}
	outputLayer.getActivation();
}

float NeuralNetwork::backwardPropagation( float *targetOutput, float *inputs, 
													float learningRate )
{
	float globalError = 0;
	float localError;
	float sum = 0;
	float csum = 0;
	float delta;
	float udelta;
	float output;

	forwardPropagation( inputs );
   
	for (int i = 0; i < outputLayer.numNeurons; ++i)
	{
		output = outputLayer.neurons[i]->neuronOut;
		localError += ( targetOutput[i]-output )*output*(1-output);
		globalError += pow( (targetOutput[i] - output), 2 );
		for (int j = 0; j < outputLayer.numLayerInputs; ++j)
		{
			delta = outputLayer.neurons[i]->deltas[j];
			udelta = learningRate*localError*outputLayer.layerInputs[i];
			outputLayer.neurons[i]->weights[j]+=udelta;
			outputLayer.neurons[i]->deltas[j] = udelta;
			sum += outputLayer.neurons[i]->weights[j]*localError;
		}
	}

	for ( int i = numHiddenLayers-1 ; i >= 0; --i )
	{
		for (int j = 0; j < hiddenLayers[i]->numNeurons; ++j)
		{
			output = hiddenLayers[i]->neurons[j]->neuronOut;
			localError = output * ( 1 - output ) * sum;
			for ( int k = 0; k < hiddenLayers[i]->numLayerInputs; ++k )
			{
				delta = hiddenLayers[i]->neurons[j]->deltas[k];
				udelta = learningRate*localError*hiddenLayers[i]->layerInputs[k];
				hiddenLayers[i]->neurons[j]->weights[k] += udelta;
				hiddenLayers[i]->neurons[j]->deltas[k] = udelta;
				csum += hiddenLayers[i]->neurons[j]->weights[k] * localError;
			}
		}
		sum = csum;
		csum = 0;
	}

	for (int i = 0; i < inputLayer.numLayerInputs; ++i)
	{
		output = inputLayer.neurons[i]->neuronOut;
		localError = output * ( 1 - output ) * sum;
		for (int j = 0; j < inputLayer.numLayerInputs; ++j)
		{
			delta = inputLayer.neurons[i]->deltas[j];
			udelta = learningRate*localError*inputLayer.layerInputs[j];
			inputLayer.neurons[i]->weights[j] += udelta;
			inputLayer.neurons[i]->deltas[j] = udelta;
		}
	}
	return globalError/2;

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

float sigmoid( float x )
{
	return 1/(1+exp(-x));
}

float gaussian( float x )
{
	return exp(-pow(x,2));
}

// int main(int argc, char const *argv[])
// {
	
// 	return 0;
// }
