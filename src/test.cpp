//
// Created by Leonard on 2025-11-18.
//

#include <iostream>

#include "cpu_only/test/math.hpp"

int main() {
    test_matrix_add();
    test_matrix_mult();
    test_matrix_mult_add();
    test_matrix_mult_add_vec();
    test_matrix_populate_by_indices();

    std::cout << "All tests succeeded!" << std::endl;
    return 0;
}
