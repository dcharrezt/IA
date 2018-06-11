#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"

class NeuralNetwork {
	private:
		Layer inputLayer;
		Layer outputLayer;
		Layer **hiddenLayers;
		int numHiddenLayers;
	public:
		int numInputs;
		int numOutputs;
		int numEpochs;

		double learningRate;

		NeuralNetwork();
		~NeuralNetwork();

		void create( int numLayerInputs, int numInputLayerNeurons, 
			int numOutputLayerNeurons, int *hiddenLayers, int numHiddenLayers);
		void forwardPropagation( float *input );
		float backwardPropagation( float *targetOutput, float *inputs, 
									float learningRate );
		
};

#endif