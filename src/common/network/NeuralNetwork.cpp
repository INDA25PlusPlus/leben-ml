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
            std::array<matrix_float_t, HIDDEN_LAYER_SIZE>, HIDDEN_LAYERS>>())
{
    constexpr matrix_float_t INPUT_SCALING = 1.0 / INPUT_LAYER_SIZE;
    input_weights->fill(INPUT_SCALING);
    for (auto i = 0; i < HIDDEN_LAYERS; i++) {
        constexpr matrix_float_t HIDDEN_SCALING = 1.0 / HIDDEN_LAYER_SIZE;
        weights->at(i).fill(HIDDEN_SCALING);
    }
}

void NeuralNetwork::forward_propagate(
    matrix_float_t const *const in,
    matrix_float_t *const out,
    size_t const m
) const {
    auto const accum = std::make_unique<matrix_float_t[]>(
        m * HIDDEN_LAYER_SIZE);
    auto a = accum.get();
    auto b = out;

    // input layer
    matrix::mult_add_vec(
        in, input_weights->data(), input_biases->data(), b,
        m, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);

    // hidden layers
    auto swapped = false;
    for (auto i = 0; i < HIDDEN_LAYERS; i++) {
        evaluation::leaky_relu(b, m, HIDDEN_LAYER_SIZE);

        std::swap(a, b);
        swapped = !swapped;

        matrix::mult_add_vec(
            a, weights->at(i).data(), biases->at(i).data(), b,
            m, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE);
    }
    // ignore all outputs of the last layer except 0-9
    evaluation::softmax(b, m, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
    if (swapped) {
        matrix::copy(accum.get(), out, m, HIDDEN_LAYER_SIZE);
    }
}
