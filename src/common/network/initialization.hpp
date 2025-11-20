//
// Created by Leonard on 2025-11-20.
//

#pragma once
#include <random>

#include "../config.hpp"


namespace initialization {
    /**
     * Initializes A using numbers generated from a normal distribution
     *
     * All matrix values are stored with the second index changing the fastest,
     * e.g. A = {a_11, a_12, a_21, a_22}.
     *
     * @param a an m × n matrix
     * @param dist the normal distribution to use for generating numbers
     * @param m the number of rows
     * @param n the number of columns
     */
    void kaiming(
        matrix_float_t *a,
        std::normal_distribution<matrix_float_t> &dist,
        size_t m,
        size_t n);
}
