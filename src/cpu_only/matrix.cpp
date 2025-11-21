//
// Created by Leonard on 2025-11-18.
//

#include "matrix.hpp"

#include <assert.h>
#include <cmath>

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
    for (auto ij = 0; ij < m * n; ij++) {
        c[ij] = a[ij] + b[ij];
    }
}

void matrix::add_all_rows_scaled(
    matrix_float_t const *const a,
    matrix_float_t const s,
    matrix_float_t *const b,
    size_t const m,
    size_t const n)
{
    for (auto j = 0; j < n; j++) {
        matrix_float_t accum = 0;
        size_t ij = j;
        for (auto i = 0; i < m; i++) {
            accum += a[ij];
            ij += n;
        }
        b[j] += accum * s;
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
                assert(!std::isnan(accum));
                assert(!std::isinf(accum));
                kj_k0 += p;
            }
            matrix_float_t const c_j = c[j];
            assert(!std::isnan(accum));
            assert(!std::isinf(accum));
            assert(!std::isnan(c_j));
            assert(!std::isinf(c_j));
            d[ij_ij] = accum + c_j;
            ij_ij++;
        }
        ik_i0 += n;
    }
}

void matrix::mult_scaled_first_t(
    matrix_float_t const *const a,
    matrix_float_t const *const b,
    matrix_float_t const s,
    matrix_float_t *const c,
    size_t const m,
    size_t const n,
    size_t const p)
{
    size_t ij_ij = 0;
    for (auto i = 0; i < m; i++) {
        for (auto j = 0; j < p; j++) {
            matrix_float_t accum = 0;
            size_t ki_k0 = 0; // index of k0 = m * k;
            size_t kj_k0 = 0; // index of k0 = p * k
            for (auto k = 0; k < n; k++) {
                matrix_float_t const a_ki = a[ki_k0 + i];
                matrix_float_t const b_kj = b[kj_k0 + j];
                accum += a_ki * b_kj;
                ki_k0 += m;
                kj_k0 += p;
            }
            c[ij_ij] = accum * s;
            ij_ij++;
        }
    }
}

void matrix::mult_second_t(
    matrix_float_t const *const a,
    matrix_float_t const *const b,
    matrix_float_t *const c,
    size_t const m,
    size_t const n,
    size_t const p)
{
    size_t ij_ij = 0;
    size_t ik_i0 = 0; // index of i0 = n * i
    for (auto i = 0; i < m; i++) {
        size_t jk_j0 = 0; // index of j0 = n * j
        for (auto j = 0; j < p; j++) {
            for (auto k = 0; k < n; k++) {
                matrix_float_t const a_ik = a[ik_i0 + k];
                matrix_float_t const b_jk = b[jk_j0 + k];
                c[ij_ij] += a_ik * b_jk;
            }
            ij_ij++;
            jk_j0 += n;
        }
        ik_i0 += n;
    }
}

void matrix::hadamard(
    matrix_float_t const *const a,
    matrix_float_t const *const b,
    matrix_float_t *const c,
    size_t const m,
    size_t const n)
{
    for (auto ij = 0; ij < m * n; ij++) {
        c[ij] = a[ij] * b[ij];
    }
}

void matrix::copy(
    matrix_float_t const *const a,
    matrix_float_t *const b,
    std::size_t const m,
    std::size_t const n)
{
    size_t ij_ij = 0;
    for (auto i = 0; i < m; i++) {
        for (auto j = 0; j < n; j++) {
            b[ij_ij] = a[ij_ij];
            ij_ij++;
        }
    }
}

void matrix::exp(
    matrix_float_t const *const a,
    matrix_float_t *const b,
    size_t const m,
    size_t const n)
{
    size_t ij = 0;
    for (auto i = 0; i < m; i++) {
        for (auto j = 0; j < n; j++) {
            auto const a_ij = a[ij];
            b[ij] = std::exp(a_ij);
            assert(!std::isnan(b[ij]));
            assert(b[ij] != 0);
            ij++;
        }
    }
}

void matrix::normalize_rows(
    matrix_float_t *const a,
    size_t const m,
    size_t const n,
    size_t const N)
{
    size_t ij = 0;
    for (auto i = 0; i < m; i++) {
        matrix_float_t sum = 0;

        size_t index = ij;
        for (auto j = 0; j < N; j++) {
            sum += a[index];
            assert(a[index] != 0);
            index++;
        }

        // if sum == 0, set scale to 1
        sum += (sum == 0);

        matrix_float_t const scale = 1.0 / sum;
        assert(!std::isnan(scale));
        assert(!std::isinf(scale));

        index = ij;
        for (auto j = 0; j < N; j++) {
            a[index] *= scale;
            assert(!std::isnan(a[index]));
            assert(!std::isinf(a[index]));
            index++;
        }

        ij += n;
    }
}

void matrix::to_normalized_float(
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
