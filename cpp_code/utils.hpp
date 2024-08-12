#ifndef PRINT_VECTOR_HPP
#define PRINT_VECTOR_HPP

#include <iostream>
#include <vector>
#include <string>

template<typename T>
void printVector(const std::vector<T>& vec, const std::string title = "Vettore: ") {
    std::cout << title;
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

#endif // PRINT_VECTOR_HPP
