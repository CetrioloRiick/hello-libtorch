#pragma once

#include <torch/torch.h>

struct MartinoNet : torch::nn::Module {
  MartinoNet();
  torch::Tensor forward(torch::Tensor x);

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
