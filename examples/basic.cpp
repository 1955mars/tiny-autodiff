#include "tiny_autodiff.h"
#include <fstream>


void runTests();

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



    Value a{3.0};
    Value b{-4.0};

    Value c = a/b + a/b;

    Value x{0.5};
    Value y{2.0};
    Value z = tanh(x * y + 1.0);

    z.backward();

    std::cout << x.grad() << "|" << y.grad() << std::endl;
    std::ofstream("graph.dot") << z.to_dot();

    */

    runTests();

}

void runTests() {

    // Identity test (sanity check of gradcheck itself)
    gradcheck([](Value x) { return x; }, 3.5);     // should give 1.0
    // Binary ops via composition with a constant
    gradcheck([](Value x) { return x + 2.0; },           3.5);     // 1.0
    gradcheck([](Value x) { return x - 2.0; },           3.5);     // 1.0
    gradcheck([](Value x) { return x * 2.0; },           3.5);     // 2.0
    gradcheck([](Value x) { return x / 2.0; },           3.5);     // 0.5
    gradcheck([](Value x) { return -x; },                3.5);     // -1.0

    // Powers via repeated multiplication
    gradcheck([](Value x) { return x * x; },             3.5);     // 2x = 7.0
    gradcheck([](Value x) { return x * x * x; },         3.5);     // 3x² = 36.75

    // tanh
    gradcheck([](Value x) { return tanh(x); },           0.5);     // 1 - tanh(0.5)² ≈ 0.7864

    // Composition — the real test
    gradcheck([](Value x) { return tanh(x * x + 1.0); }, 0.5);     // chain rule through */tanh

    // Multi-use: x appears twice
    gradcheck([](Value x) { return x + x; },             3.5);     // 2.0 — catches = vs += bugs!
    
}