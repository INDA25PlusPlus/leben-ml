//
// Created by Leonard on 2025-11-18.
//

#pragma once
#include <array>
#include <memory>

#include "../config.hpp"


constexpr size_t INPUT_WEIGHTS_COUNT = INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE;
constexpr size_t WEIGHTS_COUNT = HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE;

class NeuralNetwork {
    std::unique_ptr<std::array<matrix_float_t, INPUT_WEIGHTS_COUNT>>
        input_weights;
    std::unique_ptr<std::array<matrix_float_t, HIDDEN_LAYER_SIZE>> input_biases;

    std::unique_ptr<std::array<std::array<matrix_float_t, WEIGHTS_COUNT>,
        HIDDEN_LAYERS>> weights;
    std::unique_ptr<std::array<std::array<matrix_float_t, HIDDEN_LAYER_SIZE>,
        HIDDEN_LAYERS>> biases;

public:
    NeuralNetwork();

    /**
     * Uses the current parameters to predict a set of m outputs based on m
     * inputs.
     *
     * Assumes out is zero-initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param in an m × n matrix, with n = INPUT_LAYER_SIZE
     * @param out an m × p matrix, with p = HIDDEN_LAYER_SIZE (not
     *        OUTPUT_LAYER_SIZE)
     * @param m the number of input values to predict the output of
     */
    void forward_propagate(
        matrix_float_t const *in,
        matrix_float_t *out,
        size_t m
    ) const;
};
