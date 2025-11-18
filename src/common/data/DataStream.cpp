//
// Created by Leonard on 2025-11-16.
//

#include "DataStream.hpp"

#include <iostream>
#include <limits>


DataStream::DataStream(
    std::ifstream images_stream,
    std::ifstream labels_stream)
  : images_stream(std::move(images_stream)),
    labels_stream(std::move(labels_stream)) {}

uint32_t read_uint32(std::ifstream &stream) {
    uint8_t bytes[4];
    stream.read(reinterpret_cast<char*>(bytes), 4);
    if (stream.fail()) {
        throw std::runtime_error("Failed to read uint32");
    }
    uint32_t result = 0;
    for (auto i = 0; i < 4; i++) {
        // assume big-endian
        result |= bytes[i] << (3 - i) * 8;
    }
    return result;
}

void read_idx_header(
    std::ifstream &stream,
    const uint32_t expected_magic_number,
    const size_t num_dims,
    size_t *dims_out)
{
    const auto magic_number = read_uint32(stream);
    if (magic_number != expected_magic_number) {
        throw std::runtime_error(std::format(
            "Magic number mismatch: 0x{:x}, expected 0x{:x}",
            magic_number, expected_magic_number));
    }
    for (auto i = 0; i < num_dims; i++) {
        dims_out[i] = read_uint32(stream);
    }
}

DataStream DataStream::create(
    const std::filesystem::path &images_idx,
    const std::filesystem::path &labels_idx)
{
    // this is done to avoid the somewhat unpredictable std::ifstream move
    // constructor
    DataStream out = {std::ifstream(), std::ifstream()};

    out.images_stream.open(images_idx, std::ios::binary);
    if (out.images_stream.fail()) {
        throw std::invalid_argument("Failed to open images file");
    }

    out.labels_stream.open(labels_idx, std::ios::binary);
    if (out.labels_stream.fail()) {
        throw std::invalid_argument("Failed to open labels file");
    }

    size_t images_dims[3];
    read_idx_header(out.images_stream,
        MNIST_IMAGES_MAGIC_NUMBER, 3, images_dims);
    if (images_dims[1] != MNIST_IMAGE_WIDTH ||
        images_dims[2] != MNIST_IMAGE_WIDTH)
    {
        throw std::runtime_error("Invalid image size");
    }

    size_t labels_dims;
    read_idx_header(out.labels_stream,
        MNIST_LABELS_MAGIC_NUMBER, 1, &labels_dims);

    if (images_dims[0] != labels_dims) {
        // no of images != no of labels
        throw std::runtime_error("Number of images and labels don't match");
    }
    return out;
}

DataSet DataStream::read_up_to_n(const size_t n) {
    auto images = std::make_unique<MnistImage[]>(n);
    auto labels = std::make_unique<uint8_t[]>(n);

    size_t size = 0;
    for (; size < n; size++) {
        if (images_stream.eof() || labels_stream.eof()) {
            break;
        }
        images_stream.read(
            reinterpret_cast<char*>(images[size].data()), MNIST_IMAGE_SIZE);
        if (images_stream.fail()) {
            throw std::runtime_error("Failed to read image");
        }
        labels_stream.read(reinterpret_cast<char*>(&labels[size]), 1);
        if (labels_stream.fail()) {
            throw std::runtime_error("Failed to read label");
        }
    }
    return {size, std::move(images), std::move(labels)};
}

DataSet DataStream::read_all() {
    return read_up_to_n(std::numeric_limits<size_t>::max());
}
