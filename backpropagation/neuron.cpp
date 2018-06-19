#include "neuron.h"

Neuron::Neuron():weights(0), deltas(0), neuronOut(0), biasWeight(0)
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
    float sign=-1;//to change sign
    float random;//to get random number

	default_random_engine generator;
	uniform_real_distribution<float> distribution(-1,1);
	float tmp;
	weights = new float[ numConnections ];
	deltas = new float[ numConnections ];

	for (int i = 0; i < numConnections; ++i)
	{
		random=(float(rand()) / float(RAND_MAX))/2.f; //min 0.5
        random*=sign;
        sign*=-1;
       
		// weights[i] = distribution(generator);
		weights[i] =random;
		deltas[i] = 0; 
	}
	random=(float(rand()) / float(RAND_MAX))/2.f; //min 0.5
    random*=sign;
    sign*=-1;
	// biasWeight = distribution(generator);
	biasWeight = random;
}