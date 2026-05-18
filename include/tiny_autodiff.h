// tiny_autodiff.h — reverse-mode autodiff engine, header-only.
//
// Design (intentional; mirrors micrograd's structure):
//   * ValueImpl is the actual DAG node (data, grad, parents, _backward).
//   * Value is a value-semantic wrapper around shared_ptr<ValueImpl>.
//     Copying a Value shares the impl — same node, same gradient bucket.
//   * Operators create a new Value whose impl points at its parents and
//     whose _backward closure knows the local derivative rules.
//   * Value::backward() does a topological sort over the DAG (visited
//     keyed on the impl pointer, not the wrapper), then calls each
//     node's _backward in reverse post-order with `out.grad += local * upstream.grad`.
//     `+=` (not `=`) so multi-use nodes accumulate gradients correctly.
//
// Currently a hello-world stub — fill in the TODOs to get a working engine.

#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <vector>

namespace tinyad {

// ---------------------------------------------------------------------------
// ValueImpl — the actual DAG node. Always held by shared_ptr.
// ---------------------------------------------------------------------------
struct ValueImpl {
  double data = 0.0;
  double grad = 0.0;
  std::vector<std::shared_ptr<ValueImpl>> prev;
  std::function<void()> _backward;  // local-derivative rule for this op

  explicit ValueImpl(double d) : data(d) {}
};

// ---------------------------------------------------------------------------
// Value — value-semantic wrapper around a shared ValueImpl.
// ---------------------------------------------------------------------------
class Value {
 public:
  explicit Value(double d)
      : impl_(std::make_shared<ValueImpl>(d)) {}

  // Internal constructor used by operator overloads.
  explicit Value(std::shared_ptr<ValueImpl> impl) : impl_(std::move(impl)) {}

  double data() const { return impl_->data; }
  double grad() const { return impl_->grad; }

  // Access to the underlying impl (used by operator overloads to wire parents).
  const std::shared_ptr<ValueImpl>& impl() const { return impl_; }

  // TODO Day 2: operator+ (add this first; verify a + b backward gives grads = 1)
  // TODO Day 3: operator*, operator-, operator/, unary minus
  // TODO Day 4: backward() — topological sort + reverse traversal
  // TODO Day 5: activations (tanh, relu, sigmoid, exp, log, pow)

  friend std::ostream& operator<<(std::ostream& os, const Value& v) {
    return os << "Value(data=" << v.data() << ", grad=" << v.grad() << ")";
  }

 private:
  std::shared_ptr<ValueImpl> impl_;
};

}  // namespace tinyad
