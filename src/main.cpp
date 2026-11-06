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
#include <opencv2/opencv.hpp>

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
    for (auto &batch : *data_loader) {
      auto predictions = marti->forward(batch.data);

      // Crea il tensore one-hot con dimensione [64, 10]
      // Il cast a kFloat32 è necessario perché MSELoss richiede che
      // predizioni e target abbiano lo stesso tipo di dato (float)
      auto target_one_hot =
          torch::one_hot(batch.target, 10).to(torch::kFloat32);

      auto error = criterion(predictions, target_one_hot);
      std::cout << error << '\n';

      optimizer.zero_grad();
      error.backward();
      optimizer.step();
    }

    // auto predictions = marti->forward(x_train);
  }

  // ... [Fine del ciclo for di training] ...

  // Carichiamo il dataset di test per verificare il modello su dati non visti
  torch::data::datasets::MNIST test_dataset{
      "/home/diego/Documents/Uni/Tesi/hello-libtorch/data",
      torch::data::datasets::MNIST::Mode::kTest}; // kTest carica i 10.000
                                                  // campioni di test

  std::cout << "\nAllenamento completato. Inserisci un indice (0-9999) per "
               "visualizzare un'immagine: ";
  size_t index;
  std::cin >> index;

  // Estraiamo il singolo esempio (immagine e target)
  auto example = test_dataset.get(index);
  at::Tensor image = example.data;    // Shape: [1, 28, 28] (non normalizzata)
  at::Tensor target = example.target; // Shape: [1] (valore da 0 a 9)

  // Prepariamo l'input: applichiamo la stessa normalizzazione usata in fase di
  // train e usiamo unsqueeze(0) per aggiungere la dimensione del batch ->
  // Shape: [1, 1, 28, 28]
  at::Tensor input = image.clone().sub_(0.1307).div_(0.3081).unsqueeze(0);

  // Mettiamo la rete in modalità valutazione (disabilita eventuali Dropout o
  // BatchNorm)
  marti->eval();

  // Eseguiamo la predizione
  at::Tensor prediction = marti->forward(input);

  // Estraiamo i valori interi dai tensori
  int true_label = target.item<int>();
  int predicted_label =
      prediction.argmax(1).item<int>(); // Prende l'indice del neurone con
                                        // l'attivazione più alta

  // Dato che siamo nel terminale, stampiamo l'immagine 28x28 in ASCII art
  std::cout << "\n--- Immagine --- \n";
  at::Tensor image_squeezed = image.squeeze();
  auto img_accessor = image_squeezed.accessor<float, 2>();
  
  // ... [codice precedente fino al calcolo delle label] ...

  std::cout << "\nRisultato aspettato (Target): " << true_label;
  std::cout << "\nRisultato previsto (Modello): " << predicted_label << "\n";

  // Rimuoviamo la dimensione del canale per avere una matrice 28x28

  // Magia nera dei puntatori: creiamo l'immagine OpenCV (cv::Mat) puntando 
  // DIRETTAMENTE alla memoria allocata dal tensore di LibTorch. 
  // CV_32FC1 significa "float a 32 bit, 1 canale (grigio)".
  cv::Mat cv_img(28, 28, CV_32FC1, image_squeezed.data_ptr<float>());

  // 28x28 pixel su un monitor moderno è un francobollo. La ridimensioniamo a 280x280.
  // INTER_NEAREST disabilita l'anti-aliasing così vedi i pixel "sgranati" originali.
  cv::Mat cv_img_resized;
  cv::resize(cv_img, cv_img_resized, cv::Size(280, 280), 0, 0, cv::INTER_NEAREST);

  // Titolo della finestra con i risultati
  std::string window_title = "Target: " + std::to_string(true_label) + 
                             " | Predizione: " + std::to_string(predicted_label);

  // Mostra l'immagine e blocca l'esecuzione finché non premi un tasto sulla finestra
  cv::imshow(window_title, cv_img_resized);
  cv::waitKey(0);
}