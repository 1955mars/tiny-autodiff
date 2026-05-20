#include "tiny_autodiff.h"
#include <fstream>

int main() {

    Value a{3.0};
    std::cout << a << std::endl;

    Value b = 5 - a;
    std::cout << b << std::endl;

    Value c = -1 + a + b;
    std::cout << c << std::endl;

    Value d = 4 * c + 5 * a - 1;
    std::cout << d << std::endl;


    d.backward();


    std::ofstream("graph.dot") << d.to_dot();


}