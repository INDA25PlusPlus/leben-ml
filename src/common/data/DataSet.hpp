//
// Created by Leonard on 2025-11-17.
//

#pragma once
#include <cstdint>
#include <memory>

#include "mnist.hpp"


struct DataSet {
    const size_t size;
    const std::unique_ptr<MnistImage[]> images;
    const std::unique_ptr<uint8_t[]> labels;
};
