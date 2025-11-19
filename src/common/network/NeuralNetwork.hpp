//
// Created by Leonard on 2025-11-18.
//

#pragma once
#include <array>
#include <memory>

#include "../config.hpp"


constexpr size_t INPUT_WEIGHTS_COUNT = INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE;
constexpr size_t WEIGHTS_COUNT = HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE;

constexpr size_t BATCH_TENSOR_SIZE = BATCH_SIZE * HIDDEN_LAYER_SIZE;

class NeuralNetwork {
    using InputLayerMatrix = std::array<matrix_float_t, INPUT_WEIGHTS_COUNT>;

    using LayerMatrix = std::array<matrix_float_t, WEIGHTS_COUNT>;
    using LayerVector = std::array<matrix_float_t, HIDDEN_LAYER_SIZE>;
    using LayerBatchTensor = std::array<matrix_float_t, BATCH_TENSOR_SIZE>;

    std::unique_ptr<InputLayerMatrix> input_weights;
    std::unique_ptr<LayerVector> input_biases;

    std::unique_ptr<std::array<LayerMatrix, HIDDEN_LAYERS>> weights;
    std::unique_ptr<std::array<LayerVector, HIDDEN_LAYERS>> biases;

    // cached pre- and post-activation neuron values for the hidden layers and
    // the output layer
    std::unique_ptr<std::array<LayerBatchTensor, HIDDEN_LAYERS + 1>> pre_cache;
    std::unique_ptr<std::array<LayerBatchTensor, HIDDEN_LAYERS + 1>> post_cache;

public:
    NeuralNetwork();

    /**
     * Uses the current parameters to predict a batch of l outputs based on l
     * inputs.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param in an m × n matrix, with m = BATCH_SIZE and n = INPUT_LAYER_SIZE
     * @param input_count the number of inputs. Must be <= BATCH_SIZE
     */
    void forward_propagate(
        matrix_float_t const *in,
        size_t input_count
    ) const;

    /**
     * Returns the output of the latest forward propagation run. Only the first
     * l rows are guaranteed to be initialized, where n is the input count of
     * the latest run.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @return an m × n matrix, with m = BATCH_SIZE and n = INPUT_LAYER_SIZE.
     *         The pointer's lifetime is the same as this object.
     */
    matrix_float_t* output() const;
};
