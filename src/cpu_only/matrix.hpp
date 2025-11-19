//
// Created by Leonard on 2025-11-18.
//

#pragma once
#include <cstddef>

#include "../common/config.hpp"

namespace matrix {
    /**
     * C = AB
     *
     * Assumes C is zero-initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param b an n × p matrix
     * @param c the product AB, an m × p matrix
     * @param m the number of rows of a
     * @param n the number of columns of a, and rows of b
     * @param p the number of columns of b
     */
    void mult(
        matrix_float_t const *a,
        matrix_float_t const *b,
        matrix_float_t *c,
        std::size_t m,
        std::size_t n,
        std::size_t p);

    /**
     * C = A + B
     *
     * Assumes C is zero-initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param b an m × n matrix
     * @param c the sum A + B, an m × n matrix
     * @param m the number of rows of a
     * @param n the number of columns of a
     */
    void add(
        matrix_float_t const *a,
        matrix_float_t const *b,
        matrix_float_t *c,
        std::size_t m,
        std::size_t n);

    /**
     * D = AB + C
     *
     * D does not have to be initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param b an n × p matrix
     * @param c an m × p matrix
     * @param d an m × p matrix, AB + C
     * @param m the number of rows of a and c
     * @param n the number of columns of a, and rows of b
     * @param p the number of columns of b and c
     */
    void mult_add(
        matrix_float_t const *a,
        matrix_float_t const *b,
        matrix_float_t const *c,
        matrix_float_t *d,
        size_t m,
        size_t n,
        size_t p);

    /**
     * D = AB + C
     *
     * Sets D to the product of A and B, and adds the row vector C to each row.
     *
     * D does not have to be initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param b an n × p matrix
     * @param c a 1 × p matrix
     * @param d an m × p matrix, AB + C
     * @param m the number of rows of a
     * @param n the number of columns of a, and rows of b
     * @param p the number of columns of b and c
     */
    void mult_add_vec(
        matrix_float_t const *a,
        matrix_float_t const *b,
        matrix_float_t const *c,
        matrix_float_t *d,
        size_t m,
        size_t n,
        size_t p);

    /**
     * B = A
     *
     * Copies the elements of A into B.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param b an m × n matrix
     * @param m the number of rows
     * @param n the number of columns
     */
    void copy(
        matrix_float_t const *a,
        matrix_float_t *b,
        size_t m,
        size_t n);

    /**
     * A_ij = exp(A_ij)
     *
     * Applies the exponential function element-wise to A.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param m the number of rows
     * @param n the number of columns
     */
    void exp(
        matrix_float_t *a,
        size_t m,
        size_t n);

    /**
     * A_i = norm(A_i)
     *
     * Normalizes the rows of A, so that sum_j A_ij = 1 for all i.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param m the number of rows
     * @param n the number of columns
     */
    void normalize_rows(
        matrix_float_t *a,
        size_t m,
        size_t n);

    /**
     * B = float(A) / 255.0
     *
     * Casts all elements of A to floats and stores them in B
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param b an m × n matrix
     * @param m the number of rows of a and b
     * @param n the number of columns of a and b
     */
    void to_normalized_float(
        uint8_t const *a,
        matrix_float_t *b,
        size_t m,
        size_t n);

    /**
     * A_ij = {1 if i == indices_j, else 0}
     *
     * Populate each row of A by zeros, except a one at the index given by
     * indices.
     *
     * Assumes A is zero-initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param indices an m × 1 matrix
     * @param a an m × n matrix
     * @param m the number of rows of a
     * @param n the number of columns of a
     */
    void populate_by_indices(
        uint8_t const *indices,
        matrix_float_t *a,
        size_t m,
        size_t n);
}
