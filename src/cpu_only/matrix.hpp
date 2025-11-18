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
     * Assumes c is zero-initialized.
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
     * Assumes c is zero-initialized.
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
     * d does not have to be initialized.
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
     * d does not have to be initialized.
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
}
