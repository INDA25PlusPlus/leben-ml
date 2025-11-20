//
// Created by Leonard on 2025-11-20.
//

#include "initialization.hpp"


std::random_device device{};
std::mt19937_64 rng{device()};

void initialization::kaiming(
    matrix_float_t *const a,
    std::normal_distribution<matrix_float_t> &dist,
    size_t const m,
    size_t const n)
{
    for (auto ij = 0; ij < m * n; ij++) {
        a[ij] = dist(rng);
    }
}
