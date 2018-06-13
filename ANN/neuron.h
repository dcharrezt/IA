#ifndef NEURON_H
#define NEURON_H

#include "includes.h"

class Neuron {
	public:
		float *weights;
		float *deltas;
		float neuronOut;

		float bias = 1;
		float biasWeight;

		Neuron();
		~Neuron();
		void create( int numConnections );
};

#endif