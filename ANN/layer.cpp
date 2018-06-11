#include "layer.h"

Layer::Layer(): neurons(0), numNeurons(0), layerInputs(0), numLayerInputs(0)
{

}

Layer::~Layer()
{
	if( neurons )
	{
		for(int i = 0; i < numNeurons; i++)
		{
			delete neurons[i];
		}
		delete [] neurons;
	}
	if( layerInputs )
	{
		delete [] layerInputs;
	}
}

void Layer::create(int numLayerInputs, int numNeurons)
{
	this->numNeurons = numNeurons;
	this->numLayerInputs = numLayerInputs;
	neurons = new Neuron*[numNeurons];
	for (int i = 0; i < numNeurons; ++i)
	{
		neurons[i] = new Neuron;
		neurons[i]->create( numLayerInputs );
	}

	layerInputs = new float[numLayerInputs];
}

void Layer::getActivation( float(*activationFunction)(float) )
{
	float sum;

	for (int i = 0; i < numNeurons; ++i)
	{
		sum = 0.;
		for (int j = 0; i < numLayerInputs; ++j)
		{
			sum += neurons[i]->weights[j] * layerInputs[j];
		}
		neurons[i]->neuronOut = activationFunction( sum );
	}
}

