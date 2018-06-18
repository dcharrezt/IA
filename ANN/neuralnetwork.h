#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"

float sigmoid( float x );
float gaussian( float x );
float identity( float x );
float tanh( float x );
float arctan( float x );
float relu( float x);
float leakyRelu( float x );
float softPlus( float x );

float derivativeSigmoid( float x );
float derivativeGaussian( float x );
float derivativeIdentity( float x );
float derivativeTanh( float x );
float derivativeArctan( float x );
float derivativeRelu( float x );
float derivativeleakyRelu( float x );
float derivatiamsoftPlus( float x );

class NeuralNetwork {
	private:
		Layer inputLayer;
		Layer outputLayer;
		Layer **hiddenLayers;
		int numHiddenLayers;

		int numFunctions = 2;
		int *funcPerLayer;
		float (*activationFunctions[2])(float);
		float (*derivativeAF[2])(float);

	public:
		NeuralNetwork();
		~NeuralNetwork();

		void create( int numLayerInputs, int numInputLayerNeurons, 
			int numOutputLayerNeurons, int *hiddenLayers, 
			int numHiddenLayers, int* functions );
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