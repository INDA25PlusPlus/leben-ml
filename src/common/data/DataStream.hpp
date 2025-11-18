//
// Created by Leonard on 2025-11-16.
//

#pragma once

#include <filesystem>
#include <fstream>

#include "DataSet.hpp"


class DataStream {
    std::ifstream images_stream;
    std::ifstream labels_stream;

    DataStream(std::ifstream images_stream, std::ifstream labels_stream);

public:
    static DataStream create(
        const std::filesystem::path &images_idx,
        const std::filesystem::path &labels_idx);

    DataSet read_up_to_n(size_t n);

    DataSet read_all();
};
