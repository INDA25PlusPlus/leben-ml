//
// Created by Leonard on 2025-11-18.
//

#pragma once
#include "data/mnist.hpp"

/////////////////////
// hyperparameters //
/////////////////////

constexpr size_t HIDDEN_LAYERS = 4;
constexpr size_t HIDDEN_LAYER_SIZE = 160;

/////////////////////

constexpr size_t INPUT_LAYER_SIZE = MNIST_IMAGE_SIZE;
constexpr size_t OUTPUT_LAYER_SIZE = 10; // duh

// required for CUDA tensor cores
static_assert(INPUT_LAYER_SIZE % 8 == 0);
static_assert(HIDDEN_LAYER_SIZE % 8 == 0);

// types
using matrix_float_t = float;
