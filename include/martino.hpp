#ifndef MARTINO_HPP
#define MARTINO_HPP
#include <torch/torch.h>

namespace hlt {
struct MartinoNet : torch::nn::Module {
  MartinoNet();
  torch::Tensor forward(torch::Tensor x);

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

struct Options {
  std::string data_path{"./data"};
  int epochs{10};
  int batch_size{64};
  int workers{2};
  double learning_rate{0.001};
  bool show_stats{true};
};

Options parse_options(int argc, char **argv);

} // namespace hlt
#endif