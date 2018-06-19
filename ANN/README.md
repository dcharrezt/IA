# Backpropagation 

> Por Diego David Charrez  Ticona

**Clases**
 1.  **Neuron**
2.  **Layer**
3.   **NeuralNetwork**

**Características**
 - Se puede escoger diferentes funciones de activacion por capa.
 - Se puede Aumentar el numero de Hidden Layer.
 - Cada Hidden Layer puede tener diferente numero de neuronas y diferente función de activación.
  ![enter image description here](https://raw.githubusercontent.com/uddua/IA/master/ANN/imgs/activation_function.png)

**Compilacion**

    g++ -std=c++11 -fopenmp neuron.cpp layer.cpp neuralnetwork.cpp main.cpp

![enter image description here](https://raw.githubusercontent.com/uddua/IA/master/ANN/imgs/compile.png)

**Probado con**
 1. Iris dataset
 
![enter image description here](https://raw.githubusercontent.com/uddua/IA/master/ANN/imgs/iris.png)
 
2. XOR
 
![enter image description here](https://raw.githubusercontent.com/uddua/IA/master/ANN/imgs/xor.png)
 

