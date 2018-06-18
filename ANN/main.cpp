#include "neuralnetwork.h"

#define numTrainSet 4
#define numInputs 2
#define numNeuronsInputLayer 3
#define numNeuronsOutputLayer 1
#define numHiddenLayers 2
#define epochs 100000

enum actFunctions { Sigmoid, Gaussian, Identity, TanH, Arctan, Relu, 
							LeakyRelu, SoftPlus };

int main(int argc, char const *argv[]) {

    float x_train[numTrainSet][numInputs]=
    {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };

    float targetOutput[numTrainSet][numNeuronsOutputLayer]=
    {
        {0},
        {1},
        {1},
        {0}
    };

    NeuralNetwork net;
    float error;
    int sizesHiddenLayers[2] = {2, 2};
    int functionsPerLayer[4] = { Sigmoid, Sigmoid, Sigmoid, Sigmoid};

    net.create( numInputs, numNeuronsInputLayer, numNeuronsOutputLayer, 
    				sizesHiddenLayers, numHiddenLayers, functionsPerLayer);

    for( int i = 0; i < epochs; i++) {
        error=0;
        for(int j=0; j < numTrainSet; j++)
        {
            error+=net.backwardPropagation(targetOutput[j],x_train[j], 0.2);
        }
        error/=numTrainSet;
        cout << "ERROR:" << error << endl;
    }

    for(int i = 0; i < numTrainSet; i++) {
        net.forwardPropagation(x_train[i]);
        cout << "TESTED x_train " << i << " DESIRED OUTPUT: " << *targetOutput[i] << 
        " NET RESULT: "<< net.getOutput().neurons[0]->neuronOut << endl;
    }

	return 0;
}
