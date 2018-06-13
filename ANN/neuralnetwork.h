#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"

float sigmoid( float x );
float binaryStep( float x );
float gaussian( float x );

class NeuralNetwork {
	private:
		Layer inputLayer;
		Layer outputLayer;
		Layer **hiddenLayers;
		int numHiddenLayers;

		float (*activationFunctions[2])(float);

	public:
		NeuralNetwork();
		~NeuralNetwork();

		void create( int numLayerInputs, int numInputLayerNeurons, 
			int numOutputLayerNeurons, int *hiddenLayers, int numHiddenLayers);
		void forwardPropagation( float *input );
		float backwardPropagation( float *targetOutput, float *inputs, 
									float learningRate );
		void updateNextLayerInput( int layerIndex );

		inline Layer &getOutput()
		{
        	return outputLayer;
		}

};

#endif