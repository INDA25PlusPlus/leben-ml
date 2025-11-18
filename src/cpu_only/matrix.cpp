//
// Created by Leonard on 2025-11-18.
//

#include "matrix.hpp"

#include <cmath>

void matrix::mult(
    matrix_float_t const *const a,
    matrix_float_t const *const b,
    matrix_float_t *const c,
    std::size_t const m,
    std::size_t const n,
    std::size_t const p)
{
    // how grueling this is
    // the poor cpu so slow
    // unlike tensor cores

    std::size_t ij_ij = 0;
    for (auto i = 0; i < m; i++) {
        std::size_t const ik_i0 = n * i;
        for (auto j = 0; j < p; j++) {
            std::size_t const kj_0j = n * j;
            for (auto k = 0; k < n; k++) {
                matrix_float_t const a_ik = a[ik_i0 + k];
                matrix_float_t const b_kj = b[kj_0j + k];
                c[ij_ij] += a_ik * b_kj;
            }
            ij_ij++;
        }
    }
}

void matrix::add(
    matrix_float_t const *const a,
    matrix_float_t const *const b,
    matrix_float_t *const c,
    std::size_t const m,
    std::size_t const n)
{
    std::size_t ij_ij = 0;
    for (auto i = 0; i < m; i++) {
        for (auto j = 0; j < n; j++) {
            c[ij_ij] = a[ij_ij] + b[ij_ij];
            ij_ij++;
        }
    }
}

void matrix::mult_add(
    matrix_float_t const *const a,
    matrix_float_t const *const b,
    matrix_float_t const *const c,
    matrix_float_t *const d,
    size_t const m,
    size_t const n,
    size_t const p)
{
    std::size_t ij_ij = 0;
    for (auto i = 0; i < m; i++) {
        std::size_t const ik_i0 = n * i;
        for (auto j = 0; j < p; j++) {
            matrix_float_t accum = 0;
            std::size_t const kj_0j = n * j;
            for (auto k = 0; k < n; k++) {
                matrix_float_t const a_ik = a[ik_i0 + k];
                matrix_float_t const b_kj = b[kj_0j + k];
                accum += a_ik * b_kj;
            }
            matrix_float_t const c_ij = c[ij_ij];
            d[ij_ij] = accum + c_ij;
            ij_ij++;
        }
    }
}

void matrix::mult_add_vec(
    matrix_float_t const *const a,
    matrix_float_t const *const b,
    matrix_float_t const *const c,
    matrix_float_t *const d,
    size_t const m,
    size_t const n,
    size_t const p)
{
    std::size_t ij_ij = 0;
    for (auto i = 0; i < m; i++) {
        std::size_t const ik_i0 = n * i;
        for (auto j = 0; j < p; j++) {
            matrix_float_t accum = 0;
            std::size_t const kj_0j = n * j;
            for (auto k = 0; k < n; k++) {
                matrix_float_t const a_ik = a[ik_i0 + k];
                matrix_float_t const b_kj = b[kj_0j + k];
                accum += a_ik * b_kj;
            }
            matrix_float_t const c_j = c[j];
            d[ij_ij] = accum + c_j;
            ij_ij++;
        }
    }
}
