//
// Created by Leonard on 2025-11-18.
//

#include "evaluation.hpp"

#include <cmath>


void evaluation::activation_function(
    matrix_float_t *const a,
    size_t const m,
    size_t const n)
{
    for (auto ij = 0; ij < m * n; ij++) {
        a[ij] = std::tanh(a[ij]);
    }
}

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
