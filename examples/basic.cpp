#include "tiny_autodiff.h"
#include <fstream>


void runTests();

int main() {
/*
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

    // relu
    gradcheck([](Value x) { return relu(x); },           3.0);      // 1.0
    gradcheck([](Value x) { return relu(x); },           -3.0);     // 0.0

    // Composition through arithmetic — the real test
    gradcheck([](Value x) { return relu(x * x - 4.0); }, 3.0);  // x²-4=5>0, so deriv = 2x = 6
    gradcheck([](Value x) { return relu(x * x - 4.0); }, 1.0);  // x²-4=-3<0, so deriv = 0

    // Multi-use with unary minus — relu in both branches
    gradcheck([](Value x) { return relu(x) + relu(-x); }, 3.0);  // = 3 + 0 = 3; deriv = 1
    gradcheck([](Value x) { return relu(x) + relu(-x); }, -3.0); // = 0 + 3 = 3; deriv = -1

    //sigmoid
    // σ(0) = 0.5, σ'(0) = 0.25 — the canonical test
    gradcheck([](Value x) { return sigmoid(x) + sigmoid(x); }, 0.0);

    // Off-zero: σ(2) ≈ 0.881, σ'(2) ≈ 0.105
    gradcheck([](Value x) { return sigmoid(x); }, 2.0);

    // Negative: σ(-2) ≈ 0.119, σ'(-2) ≈ 0.105 (symmetric)
    gradcheck([](Value x) { return sigmoid(x); }, -2.0);

    // Composition through arithmetic
    gradcheck([](Value x) { return sigmoid(x * x - 1.0); }, 1.5);

    // Identity: σ(x) + σ(-x) = 1 always, so deriv = 0 at every x.
    // Cute test that exercises multi-use + symmetry.
    gradcheck([](Value x) { return sigmoid(x) + sigmoid(-x); }, 0.7);
}