//
// Created by Leonard on 2025-11-18.
//

#include "matrix.hpp"

void matrix_mult(
    matrix_float_t const *const a,
    matrix_float_t const *const b,
    matrix_float_t *const c,
    std::size_t const m,
    std::size_t const n,
    std::size_t const p)
{
    // how grueling this is
    // the poor cpu is slow
    // unlike tensor cores

    std::size_t ij_ij = 0;
    for (auto i = 0; i < m; i++) {
        const std::size_t ik_i0 = n * i;
        for (auto j = 0; j < p; j++) {
            const std::size_t kj_0j = n * j;
            for (auto k = 0; k < n; k++) {
                const matrix_float_t a_ik = a[ik_i0 + k];
                const matrix_float_t b_kj = b[kj_0j + k];
                c[ij_ij] += a_ik * b_kj;
            }
            ij_ij++;
        }
    }
}

void matrix_add(
    matrix_float_t const *const a,
    matrix_float_t const *const b,
    matrix_float_t *const c,
    std::size_t const m,
    std::size_t const p)
{
    std::size_t ij_ij = 0;
    for (auto i = 0; i < m; i++) {
        for (auto j = 0; j < p; j++) {
            c[ij_ij] = a[ij_ij] + b[ij_ij];
            ij_ij++;
        }
    }
}
