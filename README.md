# tiny-autodiff

A tiny reverse-mode automatic differentiation engine in C++ — a port and
extension of Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd), built to internalise
how autograd works under the hood.

## Status

Phase 0 — under construction.

- [x] Project skeleton (CMake, header-only library)
- [ ] `Value` class with scalar autodiff (`+`, `-`, `*`, `/`, unary `-`)
- [ ] `backward()` with topological-sort reverse traversal
- [ ] Activations (`tanh`, `relu`, `sigmoid`, `exp`, `log`, `pow`)
- [ ] Gradient check via numerical finite differences
- [ ] Tiny MLP trained on a 2D toy dataset
- [ ] (Stretch) `Tensor` extension

## Build

```bash
cmake -S . -B build
cmake --build build
./build/basic
```

## Why

Building autograd by hand is the cleanest way to internalise how every modern
ML framework actually works. The DAG construction is the same node-graph
evaluator pattern used in engine authoring tools; the new layer is per-op
derivative rules + topological reverse traversal.

## License

MIT — see [LICENSE](LICENSE).
