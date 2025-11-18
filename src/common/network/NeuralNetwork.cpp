//
// Created by Leonard on 2025-11-18.
//

#include "NeuralNetwork.hpp"

#include "../../cpu_only/evaluation.hpp"
#include "../../cpu_only/matrix.hpp"


NeuralNetwork::NeuralNetwork()
  : input_weights(
        std::make_unique<std::array<matrix_float_t, INPUT_WEIGHTS_COUNT>>()),
    input_biases(
        std::make_unique<std::array<matrix_float_t, HIDDEN_LAYER_SIZE>>()),
    weights(
        std::make_unique<std::array<
            std::array<matrix_float_t, WEIGHTS_COUNT>, HIDDEN_LAYERS>>()),
    biases(
        std::make_unique<std::array<
            std::array<matrix_float_t, HIDDEN_LAYER_SIZE>, HIDDEN_LAYERS>>()) {}

void NeuralNetwork::forward_propagate(
    matrix_float_t const *const in,
    matrix_float_t *const out,
    size_t const m
) const {
    // input layer
    matrix::mult_add_vec(
        in, input_weights->data(), input_biases->data(), out,
        m, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
    evaluation::activation_function(out, m, HIDDEN_LAYER_SIZE);

    // hidden layers
    for (auto i = 0; i < HIDDEN_LAYERS; i++) {
        matrix::mult_add_vec(
            out, weights->at(i).data(), biases->at(i).data(), out,
            m, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE);
        evaluation::activation_function(out, m, HIDDEN_LAYER_SIZE);
    }
}
