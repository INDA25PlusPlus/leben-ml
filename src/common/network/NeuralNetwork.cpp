//
// Created by Leonard on 2025-11-18.
//

#include "NeuralNetwork.hpp"

#include <assert.h>

#include "../../cpu_only/evaluation.hpp"
#include "../../cpu_only/matrix.hpp"


// skip to line 45 if you have boilerplate phobia like me
NeuralNetwork::NeuralNetwork()
  : input_weights(
        std::make_unique<
            std::array<matrix_float_t, INPUT_WEIGHTS_COUNT>>()),
    input_biases(
        std::make_unique<
            std::array<matrix_float_t, HIDDEN_LAYER_SIZE>>()),

    weights(
        std::make_unique<
            std::array<
                std::array<matrix_float_t, WEIGHTS_COUNT>,
                HIDDEN_LAYERS>>()),
    biases(
        std::make_unique<
            std::array<
                std::array<matrix_float_t, HIDDEN_LAYER_SIZE>,
                HIDDEN_LAYERS>>()),

    input_layer(nullptr),
    input_count(0),

    pre_cache(
        std::make_unique<
            std::array<
                std::array<matrix_float_t, BATCH_TENSOR_SIZE>,
                HIDDEN_LAYERS + 1>>()),
    post_cache(
        std::make_unique<
            std::array<
                std::array<matrix_float_t, BATCH_TENSOR_SIZE>,
                HIDDEN_LAYERS + 1>>())
{
    constexpr matrix_float_t INPUT_SCALING = 1.0 / INPUT_LAYER_SIZE;
    input_weights->fill(INPUT_SCALING);
    for (auto i = 0; i < HIDDEN_LAYERS; i++) {
        constexpr matrix_float_t HIDDEN_SCALING = 1.0 / HIDDEN_LAYER_SIZE;
        weights->at(i).fill(HIDDEN_SCALING);
    }
}

void NeuralNetwork::set_input(
    matrix_float_t *const input_layer,
    size_t const input_count)
{
    this->input_layer = input_layer;
    this->input_count = input_count;
}

NeuralNetwork::LayerBatchTensor const& NeuralNetwork::output() const {
    return post_cache->at(HIDDEN_LAYERS);
}

void NeuralNetwork::forward_propagate() const {
    assert(input_layer != nullptr || input_count == 0);

    // first layer
    matrix::mult_add_vec(
        input_layer,
        input_weights->data(),
        input_biases->data(),
        pre_cache->at(0).data(),
        input_count, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
    evaluation::leaky_relu(
        pre_cache->at(0).data(),
        post_cache->at(0).data(),
        input_count, HIDDEN_LAYER_SIZE);

    // hidden layers
    for (auto i = 0; i < HIDDEN_LAYERS; i++) {
        matrix::mult_add_vec(
            post_cache->at(i).data(),
            weights->at(i).data(),
            biases->at(i).data(),
            pre_cache->at(i + 1).data(),
            input_count, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE);

        if (i < HIDDEN_LAYERS - 1) {
            evaluation::leaky_relu(
                pre_cache->at(i + 1).data(),
                post_cache->at(i + 1).data(),
                input_count, HIDDEN_LAYER_SIZE);
        } else {
            // output layer

            // ignore all outputs of the last layer except 0-9
            evaluation::softmax(
                pre_cache->at(i + 1).data(),
                post_cache->at(i + 1).data(),
                input_count, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
        }
    }
}
