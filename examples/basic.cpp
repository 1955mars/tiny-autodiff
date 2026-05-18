// basic.cpp — smallest demo of the autodiff engine.
//
// Day 2 target: implement `operator+` and uncomment the block below, then
// run this binary and verify by hand:
//
//   c = a + b
//   d(c)/d(a) = 1, d(c)/d(b) = 1
//   so after c.backward(): a.grad == 1.0, b.grad == 1.0

#include "tiny_autodiff.h"
#include <iostream>

int main() {
  using tinyad::Value;

  Value a(2.0);
  Value b(-3.0);

  std::cout << "a = " << a << "\n";
  std::cout << "b = " << b << "\n";

  // -- Day 2: implement operator+ and uncomment this block --
  // auto c = a + b;
  // c.backward();
  // std::cout << "c       = " << c       << "\n";
  // std::cout << "a.grad  = " << a.grad() << "  (expected 1.0)\n";
  // std::cout << "b.grad  = " << b.grad() << "  (expected 1.0)\n";

  return 0;
}
