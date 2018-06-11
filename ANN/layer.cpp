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
	neurons = new Neuron*[numNeurons];
	for (int i = 0; i < numNeurons; ++i)
	{
		neurons[i] = new Neuron;
		neurons[i]->create( numLayerInputs );
	}

	layerInputs = new float[numLayerInputs];
	this->numNeurons = numNeurons;
	this->numLayerInputs = numLayerInputs;
}

