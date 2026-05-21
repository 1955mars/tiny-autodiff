#include "tiny_autodiff.h"
#include <fstream>

int main() {
/*
    Value a{3.0};
    std::cout << a << std::endl;

    Value b = 5 - a;
    std::cout << b << std::endl;

    Value c = -1 + a + b;
    std::cout << c << std::endl;

    Value d = 4 * c + 5 * a - 1;
    std::cout << d << std::endl;

    d.backward();

*/

    Value a{3.0};
    Value b{-4.0};

    Value c = a/b + a/b;

    c.backward();

    std::cout << b.grad() << std::endl;
    std::ofstream("graph.dot") << c.to_dot();


}