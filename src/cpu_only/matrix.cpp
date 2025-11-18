//
// Created by Leonard on 2025-11-18.
//

#include "matrix.hpp"

void matrix::mult(
    matrix_float_t const *const a,
    matrix_float_t const *const b,
    matrix_float_t *const c,
    size_t const m,
    size_t const n,
    size_t const p)
{
    // how grueling this is
    // the poor cpu so slow
    // unlike tensor cores

    size_t ij_ij = 0;
    size_t ik_i0 = 0; // index of i0 = n * i
    for (auto i = 0; i < m; i++) {
        for (auto j = 0; j < p; j++) {
            size_t kj_k0 = 0; // index of k0 = p * k
            for (auto k = 0; k < n; k++) {
                matrix_float_t const a_ik = a[ik_i0 + k];
                matrix_float_t const b_kj = b[kj_k0 + j];
                c[ij_ij] += a_ik * b_kj;
                kj_k0 += p;
            }
            ij_ij++;
        }
        ik_i0 += n;
    }
}

void matrix::add(
    matrix_float_t const *const a,
    matrix_float_t const *const b,
    matrix_float_t *const c,
    size_t const m,
    size_t const n)
{
    size_t ij_ij = 0;
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
    size_t ij_ij = 0;
    size_t ik_i0 = 0; // index of i0 = n * i
    for (auto i = 0; i < m; i++) {
        for (auto j = 0; j < p; j++) {
            matrix_float_t accum = 0;
            size_t kj_k0 = 0; // index of k0 = p * k
            for (auto k = 0; k < n; k++) {
                matrix_float_t const a_ik = a[ik_i0 + k];
                matrix_float_t const b_kj = b[kj_k0 + j];
                accum += a_ik * b_kj;
                kj_k0 += p;
            }
            matrix_float_t const c_ij = c[ij_ij];
            d[ij_ij] = accum + c_ij;
            ij_ij++;
        }
        ik_i0 += n;
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
    size_t ij_ij = 0;
    size_t ik_i0 = 0; // index of i0 = n * i
    for (auto i = 0; i < m; i++) {
        for (auto j = 0; j < p; j++) {
            matrix_float_t accum = 0;
            size_t kj_k0 = 0; // index of k0 = p * k
            for (auto k = 0; k < n; k++) {
                matrix_float_t const a_ik = a[ik_i0 + k];
                matrix_float_t const b_kj = b[kj_k0 + j];
                accum += a_ik * b_kj;
                kj_k0 += p;
            }
            matrix_float_t const c_j = c[j];
            d[ij_ij] = accum + c_j;
            ij_ij++;
        }
        ik_i0 += n;
    }
}

void matrix::to_float(
    uint8_t const *const a,
    matrix_float_t *const b,
    size_t const m,
    size_t const n)
{
    size_t ij_ij = 0;
    for (auto i = 0; i < m; i++) {
        for (auto j = 0; j < n; j++) {
            constexpr double factor = 1.0 / 255.0;
            matrix_float_t const f = a[ij_ij];
            b[ij_ij] = f * factor;
            ij_ij++;
        }
    }
}

void matrix::populate_by_indices(
    uint8_t const *const indices,
    matrix_float_t *const a,
    size_t const m,
    size_t const n)
{
    size_t ij_ij = 0;
    for (auto i = 0; i < m; i++) {
        size_t const index = indices[i];
        a[ij_ij + index] = 1;
        ij_ij += n;
    }
}
