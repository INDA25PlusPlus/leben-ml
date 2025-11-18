//
// Created by Leonard on 2025-11-17.
//

#pragma once
#include <array>
#include <format>

constexpr size_t MNIST_IMAGE_WIDTH = 28;
constexpr size_t MNIST_IMAGE_SIZE = MNIST_IMAGE_WIDTH * MNIST_IMAGE_WIDTH;

constexpr uint32_t MNIST_IMAGES_MAGIC_NUMBER = 0x00000803;
constexpr uint32_t MNIST_LABELS_MAGIC_NUMBER = 0x00000801;

using MnistImage = std::array<uint8_t, MNIST_IMAGE_SIZE>;

inline std::string image_debug_ansi_string(MnistImage &image) {
    static_assert(MNIST_IMAGE_WIDTH % 2 == 0);

    auto out = std::string();
    size_t index = 0;
    for (auto i = 0; i < MNIST_IMAGE_WIDTH; i += 2) {
        for (auto j = 0; j < MNIST_IMAGE_WIDTH; j++) {
            auto top_pixel = image[index];
            auto bot_pixel = image[index + MNIST_IMAGE_WIDTH];
            // fg color: \x1b[38;2;R;G;Bm
            // bg color: \x1b[48;2;R;G;Bm
            out.append(std::format(
                "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀",
                top_pixel, top_pixel, top_pixel,
                bot_pixel, bot_pixel, bot_pixel));

            index++;
        }
        index += MNIST_IMAGE_WIDTH;
        // reset
        out.append("\x1b[0m");
        out.append("\n");
    }
    return out;
}
