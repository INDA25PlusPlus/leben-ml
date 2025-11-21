# leben-ml

My attempt at creating a neural network library and training a network on the MNIST dataset.

The backpropagation isn't really working but I've been pulling my hair out for like 5 hours trying to find any arithmetic bugs. I guess this is what I get for coding in C++ :(

See `src/main.cpp` for more information. The current example is a binary classification example, but it can be swapped out for the MNIST classification. Note that it just doesn't work properly at all, which is why I swapped it out for the simpler example that works sometimes, depending on weight initialization (lol).

I'll probably revisit this at a later point but this is where I'll have to stop for now.

## Usage

Build using CMake. See `scripts/` for utility scripts.
