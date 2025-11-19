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
     * Multiplies A and B.
     *
     * Assumes C is zero-initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param b an n × p matrix
     * @param c the product AB, an m × p matrix
     * @param m the number of rows of A
     * @param n the number of columns of A, and rows of B
     * @param p the number of columns of B
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
     * Adds A and B.
     *
     * Assumes C is zero-initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param b an m × n matrix
     * @param c the sum A + B, an m × n matrix
     * @param m the number of rows
     * @param n the number of columns
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
     * Multiplies A and B and adds C.
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
     * @param m the number of rows of A and C
     * @param n the number of columns of A, and rows of B
     * @param p the number of columns of B and C
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
     * D = AB + rows(C)
     *
     * Multiplies A and B, and adds the row vector C to each row.
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
     * @param m the number of rows of A
     * @param n the number of columns of A, and rows of B
     * @param p the number of columns of B and C
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
     * C = s A^T B
     *
     * Multiplies A transpose and B, and scales by s.
     *
     * Assumes C is zero-initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an n × m matrix
     * @param b an n × p matrix
     * @param c the product sA^T B, an m × p matrix
     * @param s a scalar value
     * @param m the number of columns of A
     * @param n the number of rows of A and B
     * @param p the number of columns of B
     */
    void mult_scaled_first_t(
        matrix_float_t const *a,
        matrix_float_t const *b,
        matrix_float_t s,
        matrix_float_t *c,
        size_t m,
        size_t n,
        size_t p);

    /**
     * C = A B^T
     *
     * Multiplies A and B transpose.
     *
     * Assumes C is zero-initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param b an p × n matrix
     * @param c the product A B^T, an m × p matrix
     * @param m the number of columns of A
     * @param n the number of rows of A and B
     * @param p the number of columns of B
     */
    void mult_second_t(
        matrix_float_t const *a,
        matrix_float_t const *b,
        matrix_float_t *c,
        size_t m,
        size_t n,
        size_t p);

    /**
     * C = A * B (Hadamard product)
     *
     * Multiplies A and B element-wise. (C_ij = A_ij * B_ij)
     *
     * C does not have to be initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param b an m × n matrix
     * @param c the Hadamard product A and B, an m × n matrix
     * @param m the number of columns
     * @param n the number of rows
     */
    void hadamard(
        matrix_float_t const *a,
        matrix_float_t const *b,
        matrix_float_t *c,
        size_t m,
        size_t n);

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
     * B_ij = exp(A_ij)
     *
     * Applies the exponential function element-wise to A and stores the result
     * in B.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param b an m × n matrix
     * @param m the number of rows
     * @param n the number of columns
     */
    void exp(
        matrix_float_t const *a,
        matrix_float_t *b,
        size_t m,
        size_t n);

    /**
     * A_i = norm(A_i)
     *
     * Normalizes the first N columns of the rows of A, so that sum_j A_ij = 1
     * for all i.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param m the number of rows
     * @param n the number of columns
     * @param N the number of columns to perform the calculation on
     */
    void normalize_rows(
        matrix_float_t *a,
        size_t m,
        size_t n,
        size_t N);

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
     * @param m the number of rows of A and B
     * @param n the number of columns of A and B
     */
    void to_normalized_float(
        uint8_t const *a,
        matrix_float_t *b,
        size_t m,
        size_t n);

    /**
     * A_ij = {1 if i == indices_j, else 0}
     *
     * Populates each row of A by zeros, except a one at the index given by
     * indices.
     *
     * Assumes A is zero-initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param indices an m × 1 matrix
     * @param a an m × n matrix
     * @param m the number of rows of A
     * @param n the number of columns of A
     */
    void populate_by_indices(
        uint8_t const *indices,
        matrix_float_t *a,
        size_t m,
        size_t n);
}
