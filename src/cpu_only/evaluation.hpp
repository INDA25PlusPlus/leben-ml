//
// Created by Leonard on 2025-11-18.
//

#pragma once
#include "../common/config.hpp"

namespace evaluation {
    /**
     * Applies the activation function to a in place.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param m the amount of rows
     * @param n the amount of columns
     */
    void activation_function(
        matrix_float_t *a,
        size_t m,
        size_t n);

    /**
     * Applies the activation function to a in place.
     *
     * Assumes c is zero-initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param c an m × 1 matrix, the result of applying the error function
     *        row-wise to a
     * @param m the number of values to apply the activation function to
     * @param n the size of each value
     * @param correct_index the index to assign a positive eval score to. Must
     *        be <= n
     */
    void eval_function(
        matrix_float_t const *a,
        matrix_float_t *c,
        size_t m,
        size_t n,
        size_t correct_index);
}
