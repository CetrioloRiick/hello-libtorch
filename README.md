# hello-libtorch

A small LibTorch MNIST training example.

## Build

Configure with CMake and a LibTorch installation available to CMake (by default,
`/opt/libtorch` is added to the search path). CMake fetches cxxopts during
configuration.

```sh
cmake -S . -B build
cmake --build build
```

## Run

```sh
./build/main --help
./build/main --data ./data --epochs 10
```

MNIST training images are not printed unless requested. Pass a comma-separated
list of zero-based indices to display selected examples as ASCII art:

```sh
./build/main --show-images 0,3,7
```

Per-epoch training loss and accuracy are enabled by default; disable their
calculation and output with `--no-stats`.
