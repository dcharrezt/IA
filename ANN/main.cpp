#include "neuralnetwork.h"


class CSVReader
{
	std::string fileName;
	std::string delimeter;
 
public:
	CSVReader(std::string filename, std::string delm = ",") :
			fileName(filename), delimeter(delm)
	{ }
 
	// Function to fetch data from a CSV File
	std::vector<std::vector<std::string> > getData();
};
 
/*
* Parses through csv file line by line and returns the data
* in vector of vector of strings.
*/
std::vector<std::vector<std::string> > CSVReader::getData()
{
	std::ifstream file(fileName);
 
	std::vector<std::vector<std::string> > dataList;
 
	std::string line = "";
	// Iterate through each line and split the content using delimeter
	while (getline(file, line))
	{
		std::vector<std::string> vec;
		boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
		dataList.push_back(vec);
	}
	// Close the File
	file.close();
 
	return dataList;
}


// XOR parameters
// #define numTrainSet 4
// #define numInputs 2
// #define numNeuronsInputLayer 3
// #define numNeuronsOutputLayer 1
// #define numHiddenLayers 2
// #define epochs 100000

// IRIS dataset
#define numTrainSet 90
#define numInputs 4
#define numNeuronsInputLayer 4
#define numNeuronsOutputLayer 3
#define numHiddenLayers 2
#define epochs 100000
#define numTestSet 60

enum actFunctions { Sigmoid, Gaussian, Identity, TanH, Arctan, Relu, 
							LeakyRelu, SoftPlus };

int main(int argc, char const *argv[]) {

	// XOR dataset
    // float x_train[numTrainSet][numInputs]=
    // {
    //     {0,0},
    //     {0,1},
    //     {1,0},
    //     {1,1}
    // };

    // float targetOutput[numTrainSet][numNeuronsOutputLayer]=
    // {
    //     {0},
    //     {1},
    //     {1},
    //     {0}
    // };

    // IRIS dataset
    float x_train[numTrainSet][numInputs];
    float targetOutput[numTrainSet][numNeuronsOutputLayer];

	CSVReader reader("dataset/iris_x_train_60p.csv");
	std::vector<std::vector<std::string> > dataList = reader.getData();

	for( int i = 0; i < numTrainSet; i++ )
	{
		for( int j = 0; j < numInputs; j++ )
		{
			x_train[i][j] = std::stof(dataList[i][j]);
		}
	}

	CSVReader reader2("dataset/iris_y_train_60p.csv");
	std::vector<std::vector<std::string> > dataList2 = reader2.getData();

	for( int i = 0; i < numTrainSet; i++ )
	{
		for( int j = 0; j < numNeuronsOutputLayer; j++ )
		{
			targetOutput[i][j] = std::stof(dataList2[i][j]);
		}
	}

	for( int i = 0; i < numTrainSet; i++ )
	{
		for( int j = 0; j < numNeuronsOutputLayer; j++ )
		{
			std::cout << targetOutput[i][j] << ' ';
		}
		std::cout << std::endl;
	}

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


    std::cout << "Testing " << std::endl;

    float x_test[numTestSet][numInputs];
    float y_test[numTestSet][numNeuronsOutputLayer];

    int total[3] = {};
    int total_t[3] = {};

    CSVReader reader3("dataset/iris_x_test_40p.csv");
	std::vector<std::vector<std::string> > dataList3 = reader3.getData();

	for( int i = 0; i < numTestSet; i++ )
	{
		for( int j = 0; j < numInputs; j++ )
		{
			x_train[i][j] = std::stof(dataList3[i][j]);
		}
	}

	CSVReader reader4("dataset/iris_y_test_40p.csv");
	std::vector<std::vector<std::string> > dataList4 = reader4.getData();

	for( int i = 0; i < numTestSet; i++ )
	{
		for( int j = 0; j < numNeuronsOutputLayer; j++ )
		{
			y_test[i][j] = std::stof(dataList4[i][j]);
		}
	}

	float tmp;
	float tol = 1e-2;
	float c = 0;

    for(int i=0; i < numTestSet; i++) {
    	net.forwardPropagation(x_train[i]);
    	std::cout << "\nDesired " << std::endl;


    	for (int j = 0; j < numNeuronsOutputLayer; ++j)
    	{
    		if( y_test[i][j] == 1)
    			total[j] += 1;
	    	std::cout << y_test[i][j] << ' ';
    	}
    	
    	std::cout << " Output " << std::endl;


    	for (int j = 0; j < numNeuronsOutputLayer; ++j)
    	{
    		tmp = net.getOutput().neurons[j]->neuronOut;
    		if( abs(tmp - y_test[i][j]) <= tol )
    			c++;
	    	std::cout <<  tmp << ' ';
    	}

    	std::cout << "\n\n Total per class " << std::endl;

    	for (int j = 0; j < numNeuronsOutputLayer; ++j)
    	{
	    	if( y_test[i][j] == 1 && c==3 ) {
    			total_t[j]++;
    		}
	    	std::cout << total[j] << ' ';
    	}
    	c = 0;

		std::cout << "\n\n Results " << std::endl;
    	for (int j = 0; j < numNeuronsOutputLayer; ++j)
    	{
	    	std::cout << total_t[j] << ' ';
    	}

    }

    // XOR dataset
    // for(int i = 0; i < numTrainSet; i++) {
    //     net.forwardPropagation(x_train[i]);
    //     cout << "TESTED x_train " << i << " DESIRED OUTPUT: " << *targetOutput[i] << 
    //     " NET RESULT: "<< net.getOutput().neurons[0]->neuronOut << endl;
    // }


	return 0;
}
