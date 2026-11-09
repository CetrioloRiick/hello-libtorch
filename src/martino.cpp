#include "martino.hpp"

MartinoNet::MartinoNet() {
  fc1 = register_module("fc1", torch::nn::Linear(784, 64));
  fc2 = register_module("fc2", torch::nn::Linear(64, 32));
  fc3 = register_module("fc3", torch::nn::Linear(32, 10));
}

torch::Tensor MartinoNet::forward(torch::Tensor x) {
  x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
  x = torch::dropout(x, 0.5, is_training());
  x = torch::relu(fc2->forward(x));
  return fc3->forward(x);
}
