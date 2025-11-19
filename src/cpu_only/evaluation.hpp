//
// Created by Leonard on 2025-11-18.
//

#pragma once
#include "../common/config.hpp"

namespace evaluation {
    /**
     * result_i = {log(A_ij) if j = correct_index_i, else 0}
     *
     * Evaluates each row i of A by comparing it to a "correct" value given by
     * delta_jk, where j is given by correct_index_i. Assumes each row is
     * normalized (sums to 1).
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n stochastic matrix (sum_j A_ij = 1)
     * @param correct_index an m × 1 matrix
     * @param result an m × 1 matrix
     * @param m the number of rows
     * @param n the number of columns
     */
    void eval_function(
        matrix_float_t const *a,
        uint8_t const *correct_index,
        matrix_float_t *result,
        size_t m,
        size_t n);

    /**
     * Applies the leaky ReLU function to A in place.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param m the number of rows
     * @param n the number of columns
     */
    void leaky_relu(
        matrix_float_t *a,
        size_t m,
        size_t n);

    /**
     * Applies the softmax function to the first N columns of A in place, row by
     * row.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param m the number of rows
     * @param n the number of columns
     * @param N the number of columns to perform the calculation on
     */
    void softmax(
        matrix_float_t *a,
        size_t m,
        size_t n,
        size_t N);
}
