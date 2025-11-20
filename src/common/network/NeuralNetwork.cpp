//
// Created by Leonard on 2025-11-18.
//

#include "NeuralNetwork.hpp"

#include <assert.h>
#include <random>
#include <bits/unordered_map.h>

#include "initialization.hpp"
#include "../../cpu_only/evaluation.hpp"
#include "../../cpu_only/matrix.hpp"


// skip to line 59 if you have boilerplate phobia like me
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
                HIDDEN_LAYERS + 1>>()),

    delta_cache(
        std::make_unique<
            std::array<matrix_float_t, BATCH_TENSOR_SIZE>>()),
    delta_accum(
        std::make_unique<
            std::array<matrix_float_t, BATCH_TENSOR_SIZE>>()),
    grad_cache(
        std::make_unique<
            std::array<matrix_float_t, WEIGHTS_COUNT>>()),
    input_grad_cache(
        std::make_unique<
            std::array<matrix_float_t, INPUT_WEIGHTS_COUNT>>())
{
    auto input_dist = std::normal_distribution<matrix_float_t>(
        0, std::sqrt(2.0 / INPUT_LAYER_SIZE));
    initialization::kaiming(
        input_weights->data(),
        input_dist,
        INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);

    auto hidden_dist = std::normal_distribution<matrix_float_t>(
        0, std::sqrt(2.0 / HIDDEN_LAYER_SIZE));
    for (auto i = 0; i < HIDDEN_LAYERS; i++) {
        initialization::kaiming(
            weights->at(i).data(),
            hidden_dist,
            HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE);
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

    for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
        for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
            auto const z = pre_cache->at(i).at(j);
            auto const y = post_cache->at(i).at(j);
            assert(!std::isnan(z));
            assert(!std::isnan(y));
        }
    }

    // first layer
    matrix::mult_add_vec(
        input_layer,
        input_weights->data(),
        input_biases->data(),
        pre_cache->at(0).data(),
        input_count, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);

    for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
        for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
            auto const z = pre_cache->at(i).at(j);
            auto const y = post_cache->at(i).at(j);
            assert(!std::isnan(z));
            assert(!std::isnan(y));
        }
    }

    // hidden layers
    for (auto i = 0; i < HIDDEN_LAYERS; i++) {
        evaluation::leaky_relu(
            pre_cache->at(i).data(),
            post_cache->at(i).data(),
            input_count, HIDDEN_LAYER_SIZE);

        for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
            for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
                auto const z = pre_cache->at(i).at(j);
                auto const y = post_cache->at(i).at(j);
                assert(!std::isnan(z));
                assert(!std::isnan(y));
            }
        }

        matrix::mult_add_vec(
            post_cache->at(i).data(),
            weights->at(i).data(),
            biases->at(i).data(),
            pre_cache->at(i + 1).data(),
            input_count, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE);

        for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
            for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
                auto const z = pre_cache->at(i).at(j);
                auto const y = post_cache->at(i).at(j);
                assert(!std::isnan(z));
                assert(!std::isnan(y));
            }
        }
    }

    // output layer
    evaluation::softmax(
        pre_cache->at(HIDDEN_LAYERS).data(),
        post_cache->at(HIDDEN_LAYERS).data(),
        // ignore all outputs of the last layer except 0-9
        input_count, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);

    for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
        for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
            auto const z = pre_cache->at(i).at(j);
            auto const y = post_cache->at(i).at(j);
            assert(!std::isnan(z));
            assert(!std::isnan(y));
        }
    }

    for (auto i = 0; i < HIDDEN_LAYERS; i++) {
        for (auto j = 0; j < WEIGHTS_COUNT; j++) {
            auto const w = weights->at(i).at(j);
            assert(!std::isnan(w));
        }
    }
    for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
        for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
            auto const z = pre_cache->at(i).at(j);
            auto const y = post_cache->at(i).at(j);
            assert(!std::isnan(z));
            assert(!std::isnan(y));
        }
    }
}

void NeuralNetwork::back_propagate(
    matrix_float_t const step_size,
    uint8_t const *const correct_index)
const {
    //
    // this is just me trying to make sense of tensor arithmetic, feel free to
    // skip this comment
    //
    // see https://en.wikipedia.org/wiki/Backpropagation#Matrix_multiplication
    // for more information on this algorithm
    //
    // we have dimensions
    //
    // m = input_count
    // i = INPUT_LAYER_SIZE
    // n = HIDDEN_LAYER_SIZE
    // N = HIDDEN_LAYER_COUNT
    //
    //
    // we have tensors of the shapes
    //
    // input (input_layer):           m × i
    // input_weights:                 i × n
    // input_biases:                  1 × i
    //
    // pre[0..N+1] (pre_cache):       m × n
    // post[0..N+1] (post_cache):     m × n
    // weights[0..N]:                 n × n
    // biases[0..N]:                  1 × n
    //
    // delta (delta_cache):           m × n
    // grad (grad_cache):             n × n
    // input_grad (input_grad_cache): i × n
    //
    //
    // we have functions taking tensors of shapes
    //
    // eval_grad             (m × n)           -> m × n
    // softmax_derivative    (m × n)           -> m × n
    // leaky_relu_derivative (m × n)           -> m × n
    //
    // hadamard              (m × n, m × n)    -> m × n
    //
    // mult_scaled_first_t   (m × n, m × n, 1) -> n × n
    // mult_scaled_first_t   (m × i, m × n, 1) -> i × n
    // mult_second_t         (m × n, n × n)    -> m × n
    //
    //
    // given this, the algorithm is basically:
    //
    // # initialize delta_N
    // delta = hadamard(eval_grad(post[N]), softmax_derivative(pre[N]))
    //
    // # delta_(j+1) => delta_j
    // for j = (N-1)..=0
    //     grad = mult_scaled_first_t(post[j], delta, step_size / m)
    //     delta = hadamard(
    //         mult_second_t(delta, weights[j]),
    //         leaky_relu_derivative(pre[j]))
    //     weights[j] += grad
    // end
    //
    // # last but not least, input weights
    // input_grad = mult_scaled_first_t(input, delta, step_size / m)
    // input_weights += input_grad
    //

    if (input_count == 0)
        return;
    assert(input_layer != nullptr);

    for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
        for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
            auto const z = pre_cache->at(i).at(j);
            auto const y = post_cache->at(i).at(j);
            assert(!std::isnan(z));
            assert(!std::isnan(y));
        }
    }

    evaluation::eval_grad_softmax_derivative_hadamard(
        post_cache->at(HIDDEN_LAYERS).data(),
        correct_index,
        delta_cache->data(),
        input_count, HIDDEN_LAYER_SIZE);

    for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
        for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
            auto const z = pre_cache->at(i).at(j);
            auto const y = post_cache->at(i).at(j);
            assert(!std::isnan(z));
            assert(!std::isnan(y));
        }
    }

    for (auto j_reverse = 0; j_reverse < HIDDEN_LAYERS; j_reverse++) {
        auto const j = HIDDEN_LAYERS - j_reverse - 1;
        matrix::mult_scaled_first_t(
            post_cache->at(j).data(),
            delta_cache->data(),
            step_size / input_count,
            grad_cache->data(),
            HIDDEN_LAYER_SIZE, input_count, HIDDEN_LAYER_SIZE);

        for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
            for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
                auto const z = pre_cache->at(i).at(j);
                auto const y = post_cache->at(i).at(j);
                assert(!std::isnan(z));
                assert(!std::isnan(y));
            }
        }

        matrix::mult_second_t(
            delta_cache->data(),
            weights->at(j).data(),
            delta_accum->data(),
            input_count, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE);

        for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
            for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
                auto const z = pre_cache->at(i).at(j);
                auto const y = post_cache->at(i).at(j);
                assert(!std::isnan(z));
                assert(!std::isnan(y));
            }
        }

        evaluation::leaky_relu_derivative_hadamard(
            pre_cache->at(j).data(),
            delta_accum->data(),
            delta_cache->data(),
            input_count, HIDDEN_LAYER_SIZE);

        for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
            for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
                auto const z = pre_cache->at(i).at(j);
                auto const y = post_cache->at(i).at(j);
                assert(!std::isnan(z));
                assert(!std::isnan(y));
            }
        }

        matrix::add(
            weights->at(j).data(),
            grad_cache->data(),
            weights->at(j).data(),
            HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE);

        for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
            for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
                auto const z = pre_cache->at(i).at(j);
                auto const y = post_cache->at(i).at(j);
                assert(!std::isnan(z));
                assert(!std::isnan(y));
            }
        }
    }

    matrix::mult_scaled_first_t(
        input_layer,
        delta_cache->data(),
        step_size / input_count,
        input_grad_cache->data(),
        INPUT_LAYER_SIZE, input_count, HIDDEN_LAYER_SIZE);

    for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
        for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
            auto const z = pre_cache->at(i).at(j);
            auto const y = post_cache->at(i).at(j);
            assert(!std::isnan(z));
            assert(!std::isnan(y));
        }
    }

    matrix::add(
        input_weights->data(),
        input_grad_cache->data(),
        input_weights->data(),
        INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);

    for (auto i = 0; i < HIDDEN_LAYERS; i++) {
        for (auto j = 0; j < WEIGHTS_COUNT; j++) {
            auto const w = weights->at(i).at(j);
            assert(!std::isnan(w));
        }
    }
    for (auto i = 0; i < HIDDEN_LAYERS + 1; i++) {
        for (auto j = 0; j < BATCH_TENSOR_SIZE; j++) {
            auto const z = pre_cache->at(i).at(j);
            auto const y = post_cache->at(i).at(j);
            assert(!std::isnan(z));
            assert(!std::isnan(y));
        }
    }
}
