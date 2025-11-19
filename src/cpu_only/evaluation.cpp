//
// Created by Leonard on 2025-11-18.
//

#include "evaluation.hpp"

#include <cmath>

#include "matrix.hpp"


void evaluation::eval_function(
    matrix_float_t const *const a,
    matrix_float_t const *const expect,
    matrix_float_t *const result,
    size_t const m,
    size_t const n)
{
    size_t ij = 0;
    for (auto i = 0; i < m; i++) {
        for (auto j = 0; j < n; j++) {
            auto const a_ij = a[ij];
            auto const expect_ij = expect[ij];
            auto const delta = a_ij - expect_ij;
            result[i] += delta * delta;
            ij++;
        }
    }
}

void evaluation::leaky_relu(
    matrix_float_t *const a,
    size_t const m,
    size_t const n)
{
    for (auto ij = 0; ij < m * n; ij++) {
        auto const z = a[ij];
        a[ij] = std::max(LEAKY_PARAMETER * z, z);
    }
}

void evaluation::softmax(
    matrix_float_t *const a,
    size_t const m,
    size_t const n)
{
    matrix::exp(a, m, n);
    matrix::normalize_rows(a, m, n);
}
