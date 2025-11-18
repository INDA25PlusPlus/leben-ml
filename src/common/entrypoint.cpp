//
// Created by Leonard on 2025-11-16.
//

#include "entrypoint.hpp"

#include <assert.h>
#include <iostream>

#include "data/DataStream.hpp"


int entrypoint::main() {
    auto stream = DataStream::create(
        "dataset/t10k-images-idx3-ubyte",
        "dataset/t10k-labels-idx1-ubyte");
    // auto stream = DataStream::create(
    //     "dataset/test-images-idx3-ubyte",
    //     "dataset/test-labels-idx1-ubyte");
    for (auto i = 0; i < 5; i++) {
        const auto [size, images, labels] = stream.read_up_to_n(1);
        assert(size > 0);
        const std::string debug_string = image_debug_ansi_string(images[0]);
        const uint8_t label = labels[0];
        std::cout
            << std::format("{}Label: {}", debug_string, label)
            << std::endl;
    }
    return 0;
}
