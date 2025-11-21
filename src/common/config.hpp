//
// Created by Leonard on 2025-11-18.
//

#pragma once
#include "data/mnist.hpp"


// types
using matrix_float_t = float;

// constants
// constexpr size_t INPUT_LAYER_SIZE = MNIST_IMAGE_SIZE;
// constexpr size_t OUTPUT_LAYER_SIZE = 10; // duh

constexpr size_t INPUT_LAYER_SIZE = 2;
constexpr size_t OUTPUT_LAYER_SIZE = 2; // duh

/////////////////////
// hyperparameters //
/////////////////////

// constexpr size_t BATCH_SIZE = 1;
//
// constexpr size_t HIDDEN_LAYERS = 4;
// constexpr size_t HIDDEN_LAYER_SIZE = 160;
//
// constexpr matrix_float_t LEAKY_PARAMETER = 0.1;



constexpr size_t BATCH_SIZE = 1;

constexpr size_t HIDDEN_LAYERS = 1;
constexpr size_t HIDDEN_LAYER_SIZE = 2;

constexpr matrix_float_t LEAKY_PARAMETER = 0.1;

/////////////////////

// required for CUDA tensor cores
// static_assert(INPUT_LAYER_SIZE % 8 == 0);
// static_assert(HIDDEN_LAYER_SIZE % 8 == 0);
