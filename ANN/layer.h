#ifndef LAYER_H
#define LAYER_H

#include "includes.h"
#include "neuron.h"

class Layer {
	public:
		Neuron **neurons;
		int numNeurons;
		float *layerInputs;
		int numLayerInputs;
		// double(*activationFunction)(double) f;
		
}




#endif