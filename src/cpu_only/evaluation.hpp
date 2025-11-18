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
     * @param m the number of rows
     * @param n the number of columns
     */
    void activation_function(
        matrix_float_t *a,
        size_t m,
        size_t n);

    /**
     * Evaluates each row of a matrix by comparing it to the expected value.
     *
     * Assumes result is zero-initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param expect an m × n matrix, representing the expected value for each
     *        row
     * @param result an m × 1 matrix, the result of performing a row-wise
     *        evaluation with respect to the expected value
     * @param m the number of rows
     * @param n the number of columns
     */
    void eval_function(
        matrix_float_t const *a,
        matrix_float_t const *expect,
        matrix_float_t *result,
        size_t m,
        size_t n);
}
