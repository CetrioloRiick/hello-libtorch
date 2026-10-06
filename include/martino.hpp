#ifndef MARTINO_HPP
#define MARTINO_HPP

#include <torch/torch.h>

/*
model_v1 = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 3),  # 3 Neuroni in output! Uno per classe.
) */

struct Net : torch::nn::Module {
  Net();
  torch::Tensor forward(torch::Tensor x);

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
#endif