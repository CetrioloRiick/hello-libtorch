#include "martino.hpp"

#include "cxxopts.hpp"
#include <torch/data/dataloader_options.h>
#include <torch/data/datasets/mnist.h>
#include <torch/optim/adam.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  try {
    const auto options = hlt::parse_options(argc, argv);
    torch::data::datasets::MNIST dataset(
        options.data_path, torch::data::datasets::MNIST::Mode::kTrain);

    auto training_data =
        dataset.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());
    torch::data::DataLoaderOptions loader_options;
    loader_options.batch_size(static_cast<std::size_t>(options.batch_size))
        .workers(static_cast<std::size_t>(options.workers));
    auto data_loader =
        torch::data::make_data_loader(training_data, loader_options);

    hlt::MartinoNet model;
    torch::optim::Adam optimizer(
        model.parameters(), torch::optim::AdamOptions(options.learning_rate));
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
          const auto batch_count =
              static_cast<std::size_t>(batch.target.size(0));
          total_loss += loss.item<double>() * batch_count;
          total_examples += batch_count;
          correct_predictions +=
              predictions.argmax(1).eq(batch.target).sum().item<std::int64_t>();
        }
        loss.backward();
        optimizer.step();
      }

      if (options.show_stats) {
        const double mean_loss =
            total_loss / static_cast<double>(total_examples);
        const double accuracy = 100.0 * correct_predictions / total_examples;
        std::cout << "Epoch " << (epoch + 1) << '/' << options.epochs
                  << " - loss: " << mean_loss << " - accuracy: " << accuracy
                  << "%\n";
      }
    }
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
