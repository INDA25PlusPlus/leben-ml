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
     * Assumes result is zero-initialized.
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
     * Evaluates the gradient of the evaluation function with respect to the
     * output layer neuron values stored row-wise in A.
     *
     * Assumes result is zero-initialized.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n stochastic matrix (sum_j A_ij = 1)
     * @param correct_index an m × 1 matrix
     * @param result an m × n matrix, each row storing the gradient of the eval
     *        function evaluated at the corresponding row in A.
     * @param m the number of rows
     * @param n the number of columns
     */
    void eval_function_gradient(
        matrix_float_t const *a,
        uint8_t const *correct_index,
        matrix_float_t *result,
        size_t m,
        size_t n);

    /**
     * B_ij = ReLU(A_ij)
     *
     * Applies the leaky ReLU function element-wise to A and stores the result
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
    void leaky_relu(
        matrix_float_t const *a,
        matrix_float_t *b,
        size_t m,
        size_t n);

    /**
     * C_ij = ReLU'(A_ij) * B_ij
     *
     * Applies the derivative of the leaky ReLU function to A, and stores the
     * Hadamard product of the result and B in C.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param b an m × n matrix
     * @param c an m × n matrix
     * @param m the number of rows
     * @param n the number of columns
     */
    void leaky_relu_derivative_hadamard(
        matrix_float_t const *a,
        matrix_float_t const *b,
        matrix_float_t *c,
        size_t m,
        size_t n);

    /**
     * Applies the softmax function row-wise to the first N columns of A and
     * stores the result in B.
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param b an m × n matrix
     * @param m the number of rows
     * @param n the number of columns
     * @param N the number of columns to perform the calculation on. Must be <=
     *        n.
     */
    void softmax(
        matrix_float_t const *a,
        matrix_float_t *b,
        size_t m,
        size_t n,
        size_t N);

    /**
     * C_ij = eval'(A_ij) * softmax'(B_ij)
     *
     * Calculates the gradient of the eval function evaluated at A. Calculates
     * the derivative of the softmax function evaluated at B. Stores the
     * Hadamard product of those two calculations in C.
     *
     * It can be shown that when using the cross-entropy eval function and
     * softmax activation function together, the expression simplifies:
     * hadamard(eval_grad(post[N]), softmax_derivative(pre[N]))
     * becomes
     * target[N] - post[N]
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param correct_index an m × 1 matrix
     * @param result an m × n matrix
     * @param m the number of rows
     * @param n the number of columns
     */
    void eval_grad_softmax_derivative_hadamard(
        matrix_float_t const *a,
        uint8_t const *correct_index,
        matrix_float_t *result,
        size_t m,
        size_t n);
}
