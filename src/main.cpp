#include "martino.hpp"

#include <cxxopts.hpp>
#include <torch/data/dataloader_options.h>
#include <torch/data/datasets/mnist.h>
#include <torch/nn/loss.h>
#include <torch/optim/adam.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Options {
  std::string data_path{"./data"};
  std::string image_indices;
  int epochs{10};
  int batch_size{64};
  int workers{2};
  double learning_rate{0.001};
  bool show_stats{true};
};

Options parse_options(int argc, char **argv) {
  cxxopts::Options parser("hello-libtorch", "Train a small model on MNIST");
  parser.add_options()
      ("h,help", "Show this help message")
      ("data", "MNIST data directory (downloads data if needed)",
+       cxxopts::value<std::string>()->default_value("./data"))
      ("epochs", "Number of training epochs", cxxopts::value<int>()->default_value("10"))
      ("batch-size", "Training batch size", cxxopts::value<int>()->default_value("64"))
      ("workers", "Number of data-loader workers", cxxopts::value<int>()->default_value("2"))
      ("learning-rate", "Adam learning rate", cxxopts::value<double>()->default_value("0.001"))
      ("show-images", "Comma-separated MNIST training image indices to print as ASCII (for example 0,3,7)",
+       cxxopts::value<std::string>()->default_value(""))
      ("no-stats", "Disable per-epoch loss and accuracy statistics");

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
  options.image_indices = result["show-images"].as<std::string>();
  options.show_stats = !result.count("no-stats");

  if (options.epochs <= 0 || options.batch_size <= 0 || options.workers < 0 ||
      options.learning_rate <= 0.0) {
    throw std::invalid_argument("epochs, batch-size, and learning-rate must be positive; workers cannot be negative");
  }
  return options;
}

std::vector<std::size_t> parse_image_indices(const std::string &value,
                                             std::size_t dataset_size) {
  std::vector<std::size_t> indices;
  if (value.empty()) {
    return indices;
  }

  std::stringstream input(value);
  std::string token;
  while (std::getline(input, token, ',')) {
    if (token.empty()) {
      throw std::invalid_argument("--show-images expects comma-separated non-negative indices");
    }
    std::size_t parsed_chars = 0;
    const auto index = std::stoull(token, &parsed_chars);
    if (parsed_chars != token.size() || index >= dataset_size) {
      throw std::invalid_argument("MNIST image index out of range or invalid: " + token);
    }
    indices.push_back(static_cast<std::size_t>(index));
  }
  return indices;
}

void print_image(const torch::Tensor &image, std::size_t index) {
  constexpr char shades[] = " .:-=+*#%@";
  const auto pixels = image.to(torch::kCPU).to(torch::kFloat32).contiguous().view({28, 28});
  const auto accessor = pixels.accessor<float, 2>();

  std::cout << "\nMNIST image " << index << ":\n";
  for (int row = 0; row < 28; ++row) {
    for (int column = 0; column < 28; ++column) {
      const float pixel = std::clamp(accessor[row][column], 0.0f, 1.0f);
      const auto shade = static_cast<std::size_t>(pixel * (sizeof(shades) - 2));
      std::cout << shades[shade];
    }
    std::cout << '\n';
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parse_options(argc, argv);
    torch::data::datasets::MNIST dataset(
        options.data_path, torch::data::datasets::MNIST::Mode::kTrain);
    const auto image_indices =
        parse_image_indices(options.image_indices, dataset.size().value());

    for (const auto index : image_indices) {
      print_image(dataset.get(index).data, index);
    }

    auto training_data = dataset
                             .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                             .map(torch::data::transforms::Stack<>());
    torch::data::DataLoaderOptions loader_options;
    loader_options.batch_size(static_cast<std::size_t>(options.batch_size))
        .workers(static_cast<std::size_t>(options.workers));
    auto data_loader = torch::data::make_data_loader(training_data, loader_options);

    MartinoNet model;
    torch::optim::Adam optimizer(model.parameters(),
                                 torch::optim::AdamOptions(options.learning_rate));
    torch::nn::CrossEntropyLoss criterion;

    for (int epoch = 0; epoch < options.epochs; ++epoch) {
      model.train();
      double total_loss = 0.0;
      std::size_t total_examples = 0;
      std::size_t correct_predictions = 0;

      for (auto &batch : *data_loader) {
        optimizer.zero_grad();
        const auto predictions = model.forward(batch.data);
        const auto loss = criterion(predictions, batch.target);
        if (options.show_stats) {
          const auto batch_count = static_cast<std::size_t>(batch.target.size(0));
          total_loss += loss.item<double>() * batch_count;
          total_examples += batch_count;
          correct_predictions += predictions.argmax(1).eq(batch.target).sum().item<std::int64_t>();
        }
        loss.backward();
        optimizer.step();
      }

      if (options.show_stats) {
        const double mean_loss = total_loss / static_cast<double>(total_examples);
        const double accuracy = 100.0 * correct_predictions / total_examples;
        std::cout << "Epoch " << (epoch + 1) << '/' << options.epochs
                  << " - loss: " << mean_loss << " - accuracy: " << accuracy << "%\n";
      }
    }
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
