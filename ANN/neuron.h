#ifndef NEURON_H
#define NEURON_H

#include "functions.h"

class Neuron {
	public:
		float *weights;
		float *deltas;
		float neuronOut;

		Neuron();
		~Neuron();
		void create( int numInputs );
};

#endif