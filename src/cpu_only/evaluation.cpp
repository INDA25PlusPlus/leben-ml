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
    matrix_float_t *const c,
    size_t const m,
    size_t const n,
    size_t const correct_index)
{
    size_t ij = 0;
    for (auto i = 0; i < m; i++) {
        for (auto j = 0; j < n; j++) {
            auto const a_ij = a[ij];
            if (j == correct_index) {
                c[i] += a_ij;
            } else {
                c[i] -= a_ij;
            }
        }
        ij++;
    }
}
