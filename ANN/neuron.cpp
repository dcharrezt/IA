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

void Neuron::create(int numInputs )
{
	float tmp;
	weights = new float[ numInputs ];
	deltas = new float[ numInputs ];

	for (int i = 0; i < numInputs; ++i)
	{
		weights[i] = ((float) rand() / (RAND_MAX)) + 1;
		deltas[i] = 0; 
	}
}

int main(){

	return 0;
}