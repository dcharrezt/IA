#include "neuralnetwork.h"


#define PATTERN_COUNT 4
#define PATTERN_SIZE 2
#define NETWORK_INPUTNEURONS 3
#define NETWORK_OUTPUT 1
#define HIDDEN_LAYERS 0
#define EPOCHS 20000

int main(int argc, char const *argv[])
{
    float pattern[PATTERN_COUNT][PATTERN_SIZE]=
    {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };

    float desiredout[PATTERN_COUNT][NETWORK_OUTPUT]=
    {
        {0},
        {1},
        {1},
        {0}
    };


    NeuralNetwork net;//Our neural network object
    int i,j;
    float error;
    //We create the network
    net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,HIDDEN_LAYERS,HIDDEN_LAYERS);


    //Start the neural network training
    for(i=0;i<EPOCHS;i++)
    {
        error=0;
        for(j=0;j<PATTERN_COUNT;j++)
        {
            error+=net.backwardPropagation(desiredout[j],pattern[j],0.2);
        }
        error/=PATTERN_COUNT;
        //display error
        cout << "ERROR:" << error << endl;

    }

    //once trained test all patterns

    for(i=0;i<PATTERN_COUNT;i++)
    {

        net.forwardPropagation(pattern[i]);

    //display result
        cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << *desiredout[i] << " NET RESULT: "<< net.getOutput().neurons[0]->neuronOut << endl;
    }

	return 0;
}
