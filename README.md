# tiny-autodiff

A tiny reverse-mode automatic differentiation engine in C++ — a port and
extension of Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd), built to internalise
how autograd works under the hood.

## Status

- [x] Project skeleton (CMake, header-only library)
- [x] `Value` class with scalar autodiff (`+`, `-`, `*`, `/`, unary `-`)
- [x] `backward()` with topological-sort reverse traversal
- [x] Activations (`tanh`, `relu`, `sigmoid`, `exp`, `log`, `pow`)
- [x] Gradient check via numerical finite differences
- [x] Tiny MLP trained on a toy 4-point dataset
- [ ] (Stretch) `Tensor` extension

## Build

```bash
cmake -S . -B build
cmake --build build
./build/basic
```

## What's inside

Two headers, both in `include/`:

- **`tiny_autodiff.h`** — the autodiff engine.
  - `Value` — a thin handle around a shared `ValueImpl` (data + grad + parent
    list + per-op backward closure). Shared state across handles is what lets
    `c = a + b; c.backward()` write into the same `a` the user holds — and
    what lets multi-use nodes accumulate gradient correctly.
  - Operators: `+`, `-`, `*`, `/`, unary `-`, plus `tanh`, `relu`, `sigmoid`,
    `exp`, `log`, `pow`. Each installs a `_backward` closure that pushes the
    op's local derivative into its operands' grads on the reverse pass.
  - `backward()` — topological sort over the impl-pointer DAG (post-order
    DFS, `unordered_set` for dedup), then walks it in reverse calling each
    node's closure.
  - `zero_grad()`, `step(lr)` — the primitives an optimiser needs.
  - `gradcheck(f, x)` — central-difference numerical verification of any
    `Value(Value) -> Value`. The safety net.
  - `to_dot()` — emits Graphviz DOT for the full DAG, with `data` and `grad`
    in every box. Renders cleanly with `dot -Tsvg`.

- **`nn.h`** — the neural-network classes, built on `Value`.
  - `Neuron(n_in)` — `n_in` weights + 1 bias, forward = `tanh(Σ wᵢ·xᵢ + b)`.
  - `Layer(n_in, n_out)` — `n_out` neurons sharing the same input.
  - `MLP({n_in, h1, …, n_out})` — a sequence of layers piped together.
  - Each exposes `parameters()` so an optimiser can iterate all trainable
    `Value`s for `zero_grad` / `step`.

## Training demo

`examples/basic.cpp` trains an `MLP{3, 4, 4, 1}` (41 parameters) on Karpathy's
classic toy 4-point dataset:

```cpp
std::vector<std::vector<double>> X = {
    {2.0,  3.0, -1.0},
    {3.0, -1.0,  0.5},
    {0.5,  1.0,  1.0},
    {1.0,  1.0, -1.0},
};
std::vector<double> y = { 1.0, -1.0, -1.0,  1.0 };

MLP mlp({3, 4, 4, 1});

for (int epoch = 0; epoch < 100; ++epoch) {
    Value loss{0.0};
    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<Value> xs;
        for (double v : X[i]) xs.emplace_back(v);
        auto pred = mlp(xs);
        Value diff = pred[0] - y[i];
        loss = loss + diff * diff;        // MSE-style sum
    }
    loss.zero_grad();
    loss.backward();
    for (auto& p : mlp.parameters()) p.step(0.05);
}
```

Result — 100 epochs of SGD with `lr=0.05`:

```
epoch  0   loss 10.2068
epoch 10   loss 0.140385
epoch 20   loss 0.049809
epoch 50   loss 0.0159011
epoch 90   loss 0.00814303

pred = 0.958156  target =  1
pred = -0.961308 target = -1
pred = -0.958160 target = -1
pred = 0.952601  target =  1
```

`~1300× reduction` in MSE over 100 epochs. Final predictions within `0.05` of
every target.

## Gradcheck

Every operator is verified against central-difference numerical gradients. The
test suite in `examples/basic.cpp` covers `+ - * / unary- tanh relu sigmoid exp
log pow` plus compositions (`tanh(x*x+1)`, `log(exp(x))`, `exp(x)*exp(-x)`,
…) and multi-use cases (`x + x`, `relu(x) + relu(-x)`, …). Errors in the
`1e-9` to `1e-12` range across all tests.

## Why

Building autograd by hand is the cleanest way to internalise how every modern
ML framework actually works. The DAG construction is the same node-graph
evaluator pattern used in engine authoring tools; the new layer is per-op
derivative rules + topological reverse traversal. The Tensor / GPU / kernel
fusion / distributed layers in PyTorch and JAX are plumbing on top of exactly
this idea, with vectorised data and JIT-compiled kernels.

## License

MIT — see [LICENSE](LICENSE).
