//
// Created by Leonard on 2025-11-16.
//

#include <assert.h>
#include <iostream>

#include "common/config.hpp"
#include "cpu_only/evaluation.hpp"
#include "cpu_only/matrix.hpp"
#include "common/data/DataStream.hpp"
#include "common/network/NeuralNetwork.hpp"


int main() {
    // auto stream = DataStream::create(
    //     "dataset/t10k-images-idx3-ubyte",
    //     "dataset/t10k-labels-idx1-ubyte");
    //
    // auto const [size, images_int, labels]
    //     = stream.read_up_to_n(100);
    // auto const images = std::make_unique<matrix_float_t[]>(
    //     size * MNIST_IMAGE_SIZE);
    // matrix::to_normalized_float(
    //     images_int.get()->data(), images.get(),
    //     size, MNIST_IMAGE_SIZE);
    //
    // auto network = NeuralNetwork();
    // for (auto i = 0; i < 100; i++) {
    //     auto const batch_images = &images.get()[i * 1 * MNIST_IMAGE_SIZE];
    //     auto const batch_labels = &labels.get()[i * 1];
    //
    //     network.set_input(batch_images, 1);
    //     network.forward_propagate();
    //     network.back_propagate(0.01, batch_labels);
    //
    //     auto &prediction = network.output();
    //
    //     auto const eval = std::make_unique<matrix_float_t[]>(1);
    //     evaluation::eval_function(
    //         prediction.data(), batch_labels, eval.get(),
    //         1, HIDDEN_LAYER_SIZE);
    //
    //     for (auto j = 0; j < 1; j++) {
    //         // for (auto j = 0; j < 10; j++) {
    //         //     auto const p = prediction[i * HIDDEN_LAYER_SIZE + j];
    //         //     std::cout << p << " ";
    //         // }
    //         // uint32_t const label = labels.get()[j];
    //         // std::cout << std::endl << label << " " << eval[i] << std::endl;
    //         std::cout << eval[j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // return 0;





    size_t const size = 1000;
    auto const images = std::make_unique<matrix_float_t[]>(2 * size);
    auto const labels = std::make_unique<uint8_t[]>(size);
    for (auto i = 0; i < size; i++) {
        images.get()[2 * i] = 1.0f * (i % 2);
        images.get()[2 * i + 1] = 1.0f * ((i + 1) % 2);
        labels.get()[i] = (i % 2 == 0) ? 0 : 1;
    }

    auto network = NeuralNetwork();
    for (auto i = 0; i < size / BATCH_SIZE; i++) {
        auto const batch_images = &images.get()[i * HIDDEN_LAYER_SIZE * BATCH_SIZE];
        auto const batch_labels = &labels.get()[i * BATCH_SIZE];

        network.set_input(batch_images, BATCH_SIZE);
        network.forward_propagate();
        network.back_propagate(-0.001, batch_labels);

        auto &prediction = network.output();

        auto const eval = std::make_unique<matrix_float_t[]>(BATCH_SIZE);
        evaluation::eval_function(
            prediction.data(), batch_labels, eval.get(),
            BATCH_SIZE, HIDDEN_LAYER_SIZE);

        for (auto j = 0; j < BATCH_SIZE; j++) {
            // for (auto j = 0; j < 10; j++) {
            //     auto const p = prediction[i * HIDDEN_LAYER_SIZE + j];
            //     std::cout << p << " ";
            // }
            // uint32_t const label = labels.get()[j];
            // std::cout << std::endl << label << " " << eval[i] << std::endl;
            if (std::abs(eval.get()[j] + 0.693147) < 0.0001) {
                assert(false);
            }
            std::cout << eval.get()[j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;












    // size_t const size = 100;
    // auto const images = std::make_unique<matrix_float_t[]>(2 * size);
    // auto const labels = std::make_unique<uint8_t[]>(size);
    // for (auto i = 0; i < size; i++) {
    //     images.get()[2 * i] = 1.0f * (i % 2);
    //     images.get()[2 * i + 1] = 1.0f * ((i + 1) % 2);
    //     labels.get()[i] = (i % 2 == 0) ? 0 : 1;
    // }
    //
    // auto network = NeuralNetwork();
    // for (auto i = 0; i < size / BATCH_SIZE; i++) {
    //     auto const batch_images = &images.get()[i * 2 * BATCH_SIZE];
    //     auto const batch_labels = &labels.get()[i * BATCH_SIZE];
    //
    //     network.set_input(batch_images, BATCH_SIZE);
    //     network.forward_propagate();
    //     network.back_propagate(0.01, batch_labels);
    //
    //     auto &prediction = network.output();
    //
    //     auto const eval = std::make_unique<matrix_float_t[]>(BATCH_SIZE);
    //     evaluation::eval_function(
    //         prediction.data(), batch_labels, eval.get(),
    //         4, HIDDEN_LAYER_SIZE);
    //
    //     for (auto j = 0; j < BATCH_SIZE; j++) {
    //         // for (auto j = 0; j < 10; j++) {
    //         //     auto const p = prediction[i * HIDDEN_LAYER_SIZE + j];
    //         //     std::cout << p << " ";
    //         // }
    //         // uint32_t const label = labels.get()[j];
    //         // std::cout << std::endl << label << " " << eval[i] << std::endl;
    //         std::cout << eval.get()[j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // return 0;
}
