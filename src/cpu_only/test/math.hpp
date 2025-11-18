//
// Created by Leonard on 2025-11-18.
//

#pragma once
#include <complex>

#include "../matrix.hpp"
#include "../../common/config.hpp"

inline void assert_eq(matrix_float_t const x, matrix_float_t const y, size_t const i) {
    if (std::abs(x - y) > 0.001) {
        std::cout
            << std::format("Assertion failed at {}: {} != {}", i, x, y)
        << std::endl;
        exit(1);
    }
}

inline void test_matrix_add() {
    std::array<matrix_float_t, 16> const a = {
        1, 2, 3, 4,
        1, 1, 1, 1,
        0, 0, 0, 0,
        1, 0, -1, -2
    };
    std::array<matrix_float_t, 16> const b = {
        0.1, 0.1, 0.1, 0.1,
        1, 2, 3, 4,
        1, 0, 1, 0,
        -0.5, -0.5, -0.5, -0.5
    };
    std::array<matrix_float_t, 16> const expect = {
        1.1, 2.1, 3.1, 4.1,
        2, 3, 4, 5,
        1, 0, 1, 0,
        0.5, -0.5, -1.5, -2.5
    };
    std::array<matrix_float_t, 16> c = {};
    matrix::add(a.data(), b.data(), c.data(), 4, 4);
    for (auto i = 0; i < 16; i++) {
        assert_eq(expect.at(i), c.at(i), i);
    }
}

inline void test_matrix_mult() {
    std::array<matrix_float_t, 3 * 4> const a = {
        0, 1, 0, 1,
        1, 2, 3, 4,
        5, 5, 5, 5
    };
    std::array<matrix_float_t, 4 * 2> const b = {
        0.1, 0.1,
        0.2, 0.3,
        0.3, 0.5,
        0.4, 0.7,
    };
    std::array<matrix_float_t, 3 * 2> const expect = {
        0.6, 1.0,
        3.0, 5.0,
        5.0, 8.0
    };
    std::array<matrix_float_t, 3 * 2> c = {};
    matrix::mult(a.data(), b.data(), c.data(), 3, 4, 2);
    for (auto i = 0; i < 6; i++) {
        assert_eq(expect.at(i), c.at(i), i);
    }
}

inline void test_matrix_mult_add() {
    std::array<matrix_float_t, 3 * 4> const a = {
        0, 1, 0, 1,
        1, 2, 3, 4,
        5, 5, 5, 5
    };
    std::array<matrix_float_t, 4 * 2> const b = {
        0.1, 0.1,
        0.2, 0.3,
        0.3, 0.5,
        0.4, 0.7,
    };
    std::array<matrix_float_t, 3 * 2> const c = {
        -0.5, -0.9,
        -2.9, -4.9,
        -4.9, -7.9
    };
    std::array<matrix_float_t, 3 * 2> const expect = {
        0.1, 0.1,
        0.1, 0.1,
        0.1, 0.1
    };
    std::array<matrix_float_t, 3 * 2> d = {};
    matrix::mult_add(a.data(), b.data(), c.data(), d.data(), 3, 4, 2);
    for (auto i = 0; i < 6; i++) {
        assert_eq(expect.at(i), d.at(i), i);
    }
}

inline void test_matrix_mult_add_vec() {
    std::array<matrix_float_t, 3 * 4> const a = {
        0, 1, 0, 1,
        1, 2, 3, 4,
        5, 5, 5, 5
    };
    std::array<matrix_float_t, 4 * 2> const b = {
        0.1, 0.1,
        0.2, 0.3,
        0.3, 0.5,
        0.4, 0.7,
    };
    std::array<matrix_float_t, 1 * 2> const c = {
        10.0, 20.0
    };
    std::array<matrix_float_t, 3 * 2> const expect = {
        10.6, 21.0,
        13.0, 25.0,
        15.0, 28.0
    };
    std::array<matrix_float_t, 3 * 2> d = {};
    matrix::mult_add_vec(a.data(), b.data(), c.data(), d.data(), 3, 4, 2);
    for (auto i = 0; i < 6; i++) {
        assert_eq(expect.at(i), d.at(i), i);
    }
}

inline void test_matrix_populate_by_indices() {

}
