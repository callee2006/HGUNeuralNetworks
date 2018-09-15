# HGUNeuralNetworks

This is an implementaion of neural networks written for educational purposes. It includes MLP(multi-layer perceptron), AE(auto-encoder), and RBM(restricted Boltzmann machine).
It also contains CUDA code for forward propagation. But it's less efficient than CPU implementation because it was not optimized.

Prof. Injung Kim (http://deeplearning.handong.edu), School of CSEE, Handong Global University (http://www.handong.edu).



Donwload:
  git clone https://github.com/callee2006/HGUNeuralNetworks

Compile:

  make                // using CPU
  
  make CUDA=ENABLE    // enable GPU  (less efficient than CPU)
 
