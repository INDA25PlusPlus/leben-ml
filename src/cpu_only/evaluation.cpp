//
// Created by Leonard on 2025-11-18.
//

#include "evaluation.hpp"

#include <assert.h>
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
        assert(!std::isnan(result[i]));
        ij += n;
    }
}

void evaluation::eval_function_gradient(
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
        result[i] = 1 / a_ij;
        assert(!std::isnan(result[i]));
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

void evaluation::leaky_relu_derivative_hadamard(
    matrix_float_t const *const a,
    matrix_float_t const *const b,
    matrix_float_t *const c,
    size_t const m,
    size_t const n)
{
    for (auto ij = 0; ij < m * n; ij++) {
        auto const z = a[ij];
        // 1 if z >= 0 else alpha
        auto const result = (z >= 0) + (z < 0) * LEAKY_PARAMETER;
        c[ij] = result * b[ij];
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

void evaluation::eval_grad_softmax_derivative_hadamard(
    matrix_float_t const *const a,
    uint8_t const *const correct_index,
    matrix_float_t *const result,
    size_t const m,
    size_t const n)
{
    size_t ij = 0;
    for (auto i = 0; i < m; i++) {
        auto const expected_index = correct_index[i];
        for (auto j = 0; j < n; j++) {
            matrix_float_t const expect = (j == expected_index);
            result[ij] = expect - a[ij];
            ij++;
        }
    }
}
