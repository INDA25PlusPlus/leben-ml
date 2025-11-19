//
// Created by Leonard on 2025-11-18.
//

#include "evaluation.hpp"

#include <cmath>

#include "matrix.hpp"


void evaluation::eval_function(
    matrix_float_t const *const a,
    uint8_t const *const correct_index,
    matrix_float_t *const result,
    size_t const m,
    size_t const n)
{
    size_t ij = 0;
    for (auto i = 0; i < m; i++) {
        size_t const index = correct_index[i];
        matrix_float_t const a_ij = a[ij + index];
        result[i] = std::log(a_ij);
        ij += n;
    }
}

void evaluation::leaky_relu(
    matrix_float_t const *const a,
    matrix_float_t *const b,
    size_t const m,
    size_t const n)
{
    for (auto ij = 0; ij < m * n; ij++) {
        auto const z = a[ij];
        // z if z >= 0 else alpha * z
        b[ij] = std::max(LEAKY_PARAMETER * z, z);
    }
}

void evaluation::softmax(
    matrix_float_t const *const a,
    matrix_float_t *const b,
    size_t const m,
    size_t const n,
    size_t const N)
{
    matrix::exp(a, b, m, n);
    matrix::normalize_rows(b, m, n, N);
}
