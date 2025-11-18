//
// Created by Leonard on 2025-11-18.
//

#pragma once
#include <cstddef>

#include "../common/config.hpp"

namespace matrix {
    /**
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
}
