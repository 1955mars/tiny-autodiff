#include "nn.h"

#include <fstream>


void runTests();
void testNeuron();
void testLayer();
void testMLP();

int main() {
/*
    std::ofstream("graph.dot") << z.to_dot();
    runTests();
    testNeuron();
    testLayer();
*/
    testMLP();
}

void testMLP() {
    MLP mlp{{2, 3, 1}};
    std::vector<Value> inputs{1.0, 2.0};

    auto outs = mlp(inputs);
    assert(outs.size() == 1);

    Value loss = outs[0];
    loss.zero_grad();
    loss.backward();

    auto params = mlp.parameters();
    std::cout << "params: " << params.size() << "  (expected 13)\n";
    for (auto& p : params) {
        std::cout << "  data=" << p.get() << "  grad=" << p.grad() << "\n";
    }

    std::ofstream("graph.dot") << loss.to_dot();
}


void testLayer() {
    Layer l{3, 2};
    std::vector<Value> inputs{0.1, 0.2, 0.3};

    auto outs = l(inputs);
    assert(outs.size() == 2);

    //Build a dummy scalar loss so we can backward
    Value loss = outs[0] + outs[1];
    loss.zero_grad();
    loss.backward();


    auto params = l.parameters();
    assert(params.size() == 8);

    std::cout << "param count: " << params.size() << "\n";
    for (auto& p : params) {
        std::cout << "  data=" << p.get() << "  grad=" << p.grad() << "\n";
    }

    std::ofstream("graph.dot") << loss.to_dot();
}

void testNeuron() {
    // Single Neuron
    {
        Neuron n{1};
        std::vector<Value> inputs{Value{0.5}};
        Value y = n(inputs);
        std::cout << y.get() << std::endl;

        y.zero_grad();
        y.backward();

        std::ofstream("graph.dot") << y.to_dot();

        auto params = n.parameters();
        for(auto& p : params) {
            std::cout << "param data = " << p.get() << " grad = " << p.grad() << "\n";

        }
    }

    // Three Neurons
    {
        Neuron n{3};
        std::vector<Value> inputs{0.1, 0.2, 0.3};
        Value y = n(inputs);
        std::cout << y.get() << std::endl;

        y.zero_grad();
        y.backward();

        std::ofstream("graph.dot") << y.to_dot();

        auto params = n.parameters();
        assert(params.size() == 4);
        for(auto& p : params) {
            std::cout << "param data = " << p.get() << " grad = " << p.grad() << "\n";
        }
    }
    

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

    // exp(0) = 1, exp'(0) = 1
    gradcheck([](Value x) { return exp(x); }, 0.0);

    // exp(1) ≈ 2.718, exp'(1) = exp(1) ≈ 2.718
    gradcheck([](Value x) { return exp(x); }, 1.0);

    // exp(-2) ≈ 0.135, exp'(-2) ≈ 0.135
    gradcheck([](Value x) { return exp(x); }, -2.0);

    // Composition: d/dx[exp(x²)] = 2x·exp(x²)
    // At x=1.5: x²=2.25, exp(2.25)≈9.488, deriv = 3·9.488 ≈ 28.46
    gradcheck([](Value x) { return exp(x * x); }, 1.5);

    // Composition with subtraction: exp(x - 1) at x=2 → exp(1) ≈ 2.718, deriv = exp(1)
    gradcheck([](Value x) { return exp(x - 1.0); }, 2.0);

    // Multi-use: exp(x) + exp(x) — should give 2·exp(x); deriv = 2·exp(x)
    // At x=1: analytical = 2·e ≈ 5.437
    gradcheck([](Value x) { return exp(x) + exp(x); }, 1.0);

    // The cute one: exp(x) * exp(-x) = 1 (constant), so deriv = 0
    gradcheck([](Value x) { return exp(x) * exp(-x); }, 0.7);

    // log(1) = 0, log'(1) = 1
    gradcheck([](Value x) { return log(x); }, 1.0);

    // log(e) = 1, log'(e) = 1/e ≈ 0.368
    gradcheck([](Value x) { return log(x); }, 2.71828);

    // log(2) ≈ 0.693, log'(2) = 0.5
    gradcheck([](Value x) { return log(x); }, 2.0);

    // log(0.5) ≈ -0.693, log'(0.5) = 2.0
    gradcheck([](Value x) { return log(x); }, 0.5);

    // Composition: d/dx[log(x²)] = 2/x. At x=3: deriv = 0.667
    gradcheck([](Value x) { return log(x * x); }, 3.0);

    // Cute identity: log(exp(x)) = x, so derivative = 1 for every x
    gradcheck([](Value x) { return log(exp(x)); }, 1.7);

    // Inverse identity: log(1/x) = -log(x), deriv = -1/x. At x=2: -0.5
    gradcheck([](Value x) { return log(Value{1.0} / x); }, 2.0);

    // Multi-use: log(x) + log(x) = 2·log(x), deriv = 2/x. At x=3: 0.667
    gradcheck([](Value x) { return log(x) + log(x); }, 3.0);

    // x² at x=3: deriv = 2x = 6
    gradcheck([](Value x) { return pow(x, 2.0); }, 3.0);

    // x³ at x=2: deriv = 3x² = 12
    gradcheck([](Value x) { return pow(x, 3.0); }, 2.0);

    // sqrt(x) = x^0.5 at x=4: deriv = 0.5·x^(-0.5) = 0.25
    gradcheck([](Value x) { return pow(x, 0.5); }, 4.0);

    // x^(-1) = 1/x at x=2: deriv = -1/x² = -0.25
    gradcheck([](Value x) { return pow(x, -1.0); }, 2.0);

    // x^0 = 1 (constant) at any x: deriv = 0
    gradcheck([](Value x) { return pow(x, 0.0); }, 2.5);

    // Composition: (x+1)² at x=2 → 9, deriv = 2(x+1) = 6
    gradcheck([](Value x) { return pow(x + 1.0, 2.0); }, 2.0);

    // Multi-use: x² + x² = 2x², deriv = 4x at x=3 → 12
    gradcheck([](Value x) { return pow(x, 2.0) + pow(x, 2.0); }, 3.0);

    // Minimize f(w) = (w-3)^2 by gradient descent. w should converge to 3
    {
        Value w{0.0};
        double lr = 0.1;
        for(int i=0; i<50; i++) {
            Value loss = pow(w - 3.0, 2.0);
            loss.zero_grad();
            loss.backward();
            w.step(lr);
        }
        std::cout << "w converged to " << w.get() << "  (expected ~3.0)\n";
        std::cout << "final loss " << pow(w - 3.0, 2.0).get() << "  (expected ~0)\n";
    }
}