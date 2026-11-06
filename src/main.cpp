#include "martino.hpp"
#include <ATen/core/TensorBody.h>
#include <iostream>
#include <memory>
#include <torch/data/dataloader/base.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/datasets/base.h>
#include <torch/data/datasets/mnist.h>
#include <torch/nn/modules/loss.h>
#include <torch/optim/adam.h>
#include <torch/torch.h>
#include <vector>

int main() {
  /* auto data_loader =
      torch::data::make_data_loader(torch::data::datasets::MNIST("./data").map(
                                        torch::data::transforms::Stack<>()),
                                    64); */
  // Creo un oggetto MNIST che è un Dataset, ora da quanto ho capito i dataset
  // servono principalmente come contenitoroni per cacciarci fuori tutto quello
  // che vogliamo (principalemente tensori)
  torch::data::datasets::MNIST pippo{
      "/home/diego/Documents/Uni/Tesi/hello-libtorch/data",
      torch::data::datasets::MNIST::Mode::kTrain}; // NON SO COSA SIA KTRAIN

  auto pluto =
      pippo.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::
                   Stack<>()); // Questa cosa è fondamentale per far arrivare
                               // dentro il ciclo con un tensor e non un example

  // DataLoader fa i batches per l'allenamento, ma per usare data loader bisogna
  // crearlo con make_blabla (ha bisogno del dataset chiaramente e puoi dargli
  // una apposita classe con le opzioni che vuoi modificare di data loader)
  torch::data::DataLoaderOptions options{64};
  options.batch_size(64).workers(
      4); // Qui dentro c'è una roba un po' crazy in pratica ogni funzione
          // restituisce una reference allo stessa istanza della classe e posso
          // scrivere su una sola riga + opzioni (inutile)
  auto data_loader = torch::data::make_data_loader(
      pluto,
      options); // usiamo questa sintassi perchè così crea già lui il puntatore
  // Dataloader è arrayabile sui batch che a questo punto verranno rimescolati
  // ad ogni epoca

  auto marti = std::make_shared<MartinoNet>();

  torch::optim::Adam optimizer{marti->parameters(), 0.1};
  torch::nn::MSELoss criterion{};
  for (int i{0}; i != 10; ++i) {
    marti->train();
    for (auto &batch : *data_loader) { // Quindi batch è una reference a quello
                                       // che punta data_loader (crazy)

      auto predictions = marti->forward(batch.data);
      auto error = criterion(predictions, batch.target);
      std::cout << error;
      optimizer.zero_grad();
      error.backward();
      optimizer.step();
    }

    // auto predictions = marti->forward(x_train);
  }
}