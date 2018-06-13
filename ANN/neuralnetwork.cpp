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
		int numOutputLayerNeurons, int *sizesHiddenLayers, int numHiddenLayers,
		int* functions )
{

    funcPerLayer = new int[numHiddenLayers+1];
	memcpy( funcPerLayer, functions, (numHiddenLayers+1)*sizeof(int));
	float (*activationFunctions[2])(float) = {sigmoid, gaussian};


	cout << " asd" << endl;
    for (int i = 0; i < 4; ++i)
    {
    	cout << functions[i] << endl;
    }

	inputLayer.create( numLayerInputs, numInputLayerNeurons, 
			activationFunctions[functions[0]]);
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
						 sizesHiddenLayers[i], activationFunctions[functions[i+1]] );
			}
			else
			{
				this->hiddenLayers[i]->create( sizesHiddenLayers[i-1], 
					sizesHiddenLayers[i], activationFunctions[functions[i+1]] );
			}
		}
		outputLayer.create( sizesHiddenLayers[numHiddenLayers-1], 
		numOutputLayerNeurons , activationFunctions[functions[numHiddenLayers+1]]);
	}
	else
		outputLayer.create( numInputLayerNeurons, numOutputLayerNeurons, 
										activationFunctions[functions[1]] );
}

void NeuralNetwork::forwardPropagation( float *input )
{
	memcpy( inputLayer.layerInputs, input, inputLayer.numLayerInputs*sizeof(float));
	inputLayer.getActivation();
	updateNextLayerInput(-1);
	if( hiddenLayers ) {
		for (int i = 0; i < numHiddenLayers; ++i) {
			hiddenLayers[i]->getActivation();
			updateNextLayerInput(i);
		}
	}
	outputLayer.getActivation();
}

float NeuralNetwork::backwardPropagation( float *targetOutput, float *inputs, 
													float learningRate )
{
	float globalError = 0.;
	float localError = 0.;
	float sum = 0.;
	float csum = 0.;
	float delta;
	float udelta;
	float output;

	float (*derivativeAF[2])(float) = {derivativeSigmoid, derivativeGaussian};
	forwardPropagation( inputs );

   
	for (int i = 0; i < outputLayer.numNeurons; ++i)
	{
		output = outputLayer.neurons[i]->neuronOut;
		localError = ( targetOutput[i] - output ) * (*derivativeAF[funcPerLayer[0]])(output);
		globalError += (targetOutput[i] - output) * (targetOutput[i] - output) ;
		for (int j = 0; j < outputLayer.numLayerInputs; ++j)
		{
			udelta = learningRate * localError * outputLayer.layerInputs[j];
			outputLayer.neurons[i]->weights[j] += udelta;
			outputLayer.neurons[i]->deltas[j] = udelta;
			sum += outputLayer.neurons[i]->weights[j]*localError;
		}
		outputLayer.neurons[i]->biasWeight += learningRate*localError*
										outputLayer.neurons[i]->bias;
	}

	for ( int i = (numHiddenLayers-1) ; i >= 0; i-- )
	{
		for (int j = 0; j < hiddenLayers[i]->numNeurons; ++j)
		{
			output = hiddenLayers[i]->neurons[j]->neuronOut;
			localError = (*derivativeAF[funcPerLayer[i+1]])(output) * sum;
			for ( int k = 0; k < hiddenLayers[i]->numLayerInputs; ++k ) {
				udelta = learningRate*localError*hiddenLayers[i]->layerInputs[k];
				hiddenLayers[i]->neurons[j]->weights[k] += udelta;
				hiddenLayers[i]->neurons[j]->deltas[k] = udelta;
				csum += hiddenLayers[i]->neurons[j]->weights[k] * localError;
			}

			hiddenLayers[i]->neurons[j]->biasWeight += learningRate*
								localError*hiddenLayers[i]->neurons[j]->bias;
		}
		sum = csum;
		csum = 0;
	}

	for (int i = 0; i < inputLayer.numNeurons; ++i)
	{

		output = inputLayer.neurons[i]->neuronOut;
		localError =  (*derivativeAF[funcPerLayer[numHiddenLayers+1]])(output) * sum;
		for (int j = 0; j < inputLayer.numLayerInputs; ++j)
		{
			delta = inputLayer.neurons[i]->deltas[j];
			udelta = learningRate*localError*inputLayer.layerInputs[j];
			inputLayer.neurons[i]->weights[j] += udelta;
			inputLayer.neurons[i]->deltas[j] = udelta;
		}
		inputLayer.neurons[i]->biasWeight += learningRate*localError*
							inputLayer.neurons[i]->bias; 
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
	return 1./(1.+exp(-1.*x));
}

float gaussian( float x )
{
	return exp(-pow(x,2));
}

float derivativeSigmoid( float x )
{
	return x*( 1 - x );
}

float derivativeGaussian( float x )
{
	return -2*x*exp(-(x*x));
}