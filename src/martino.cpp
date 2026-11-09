#include "martino.hpp"
#include "cxxopts.hpp"
#include <torch/torch.h>

namespace hlt {

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

Options parse_options(int argc, char **argv) {
  cxxopts::Options parser("hello-libtorch", "Train a small model on MNIST");
  parser.add_options()("h,help", "Show this help message")(
      "data", "MNIST data directory (downloads data if needed)",
      cxxopts::value<std::string>()->default_value("./data"))(
      "epochs", "Number of training epochs",
      cxxopts::value<int>()->default_value("10"))(
      "batch-size", "Training batch size",
      cxxopts::value<int>()->default_value("64"))(
      "workers", "Number of data-loader workers",
      cxxopts::value<int>()->default_value("2"))(
      "learning-rate", "Adam learning rate",
      cxxopts::value<double>()->default_value("0.001"))(
      "no-stats", "Disable per-epoch loss and accuracy statistics");

  const auto result = parser.parse(argc, argv);
  if (result.count("help")) {
    std::cout << parser.help() << '\n';
    std::exit(0);
  }

  Options options;
  options.data_path = result["data"].as<std::string>();
  options.epochs = result["epochs"].as<int>();
  options.batch_size = result["batch-size"].as<int>();
  options.workers = result["workers"].as<int>();
  options.learning_rate = result["learning-rate"].as<double>();
  options.show_stats = !result.count("no-stats");

  if (options.epochs <= 0 || options.batch_size <= 0 || options.workers < 0 ||
      options.learning_rate <= 0.0) {
    throw std::invalid_argument("epochs, batch-size, and learning-rate must be "
                                "positive; workers cannot be negative");
  }
  return options;
}
} // namespace hlt
