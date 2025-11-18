//
// Created by Leonard on 2025-11-18.
//

#pragma once
#include <cstddef>

#include "../common/config.hpp"

/**
 * All matrix values are stored with the second index changing the fastest, e.g.
 * A = {a_11, a_12, a_21, a_22}. Assumes c is zero-initialized.
 * @param a an m × n matrix
 * @param b an n × p matrix
 * @param c the product AB, an m × p matrix
 */
void matrix_mult(
    matrix_float_t const *a,
    matrix_float_t const *b,
    matrix_float_t *c,
    std::size_t m,
    std::size_t n,
    std::size_t p);

/**
 * All matrix values are stored with the second index changing the fastest, e.g.
 * A = {a_11, a_12, a_21, a_22}. Assumes c is zero-initialized.
 * @param a an m × p matrix
 * @param b an m × p matrix
 * @param c the sum A + B, an m × p matrix
 */
void matrix_add(
    matrix_float_t const *a,
    matrix_float_t const *b,
    matrix_float_t *c,
    std::size_t m,
    std::size_t p);
