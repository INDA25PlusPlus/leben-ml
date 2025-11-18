//
// Created by Leonard on 2025-11-16.
//

#include <iostream>

#include "common/config.hpp"
#include "cpu_only/evaluation.hpp"
#include "cpu_only/matrix.hpp"
#include "common/data/DataStream.hpp"
#include "common/network/NeuralNetwork.hpp"


int main() {
    auto stream = DataStream::create(
        "dataset/t10k-images-idx3-ubyte",
        "dataset/t10k-labels-idx1-ubyte");

    auto const [size, images_int, labels]
        = stream.read_up_to_n(10000);
    auto const images = std::make_unique<matrix_float_t[]>(
        size * MNIST_IMAGE_SIZE);
    matrix::to_normalized_float(
        images_int.get()->data(), images.get(),
        size, MNIST_IMAGE_SIZE);

    auto const network = NeuralNetwork();
    auto const prediction = std::make_unique<matrix_float_t[]>(
        size * HIDDEN_LAYER_SIZE);
    network.forward_propagate(images.get(), prediction.get(), size);

    auto const expect = std::make_unique<matrix_float_t[]>(
        size * MNIST_IMAGE_SIZE);
    matrix::populate_by_indices(
        labels.get(), expect.get(),
        size, HIDDEN_LAYER_SIZE);

    auto const eval = std::make_unique<matrix_float_t[]>(size);
    evaluation::eval_function(
        prediction.get(), expect.get(), eval.get(),
        size, HIDDEN_LAYER_SIZE);

    for (auto i = 0; i < size; i++) {
        std::cout << eval[i] << std::endl;
    }
    return 0;
}
