#include "neuron.h"

Neuron::Neuron():weights(0), deltas(0), neuronOut(0)
{

}

Neuron::~Neuron()
{
	if( weights )
		delete [] weights;
	if( deltas )
		delete [] deltas;
}

void Neuron::create(int numConnections )
{
	float tmp;
	weights = new float[ numConnections ];
	deltas = new float[ numConnections ];

	for (int i = 0; i < numConnections; ++i)
	{
		weights[i] = ((float) rand() / (RAND_MAX)) + 1;
		deltas[i] = 0; 
	}
}