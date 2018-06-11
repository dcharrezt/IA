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

		float (*activationFunction)(float);

		Layer();
		~Layer();
		void create(int numLayerInputs, int numNeurons, 
										float (*activationFunction)(float));
		void getActivation();

};

#endif