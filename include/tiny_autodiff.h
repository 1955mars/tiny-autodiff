#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <functional>
#include <unordered_set>
#include <cmath>

struct ValueImpl{
    double data = 0.0;
    double grad = 0.0;
    std::string op;
    std::vector<std::shared_ptr<ValueImpl>> prev;
    std::function<void()> backward;



    void to_dot_impl(std::ostringstream& os) const {
        const void* this_id = static_cast<const void*>(this);
        os << "  \"" << this_id << "\" [label=\"data=" << data << "|grad=" << grad << "\", shape=record];\n";
        if(!op.empty()) {
            os << "  \"op_" << this_id << "\" [label=\"" << op << "\", shape=circle, style=filled, fillcolor=lightgray];\n";
            os << "  \"op_" << this_id << "\" -> \"" << this_id << "\";\n";
            for (const auto& p : prev) {
                p->to_dot_impl(os);
                const void* p_id = static_cast<const void*>(p.get());
                os << "  \"" << p_id << "\" -> \"op_" << this_id << "\";\n";
            }
        }
    }

};

class Value {
public:
    Value(double val):impl(std::make_shared<ValueImpl>()) { impl->data = val;}

    void backward() {

        std::vector<ValueImpl*> topo;
        std::unordered_set<const ValueImpl*> visited;

        std::function<void(ValueImpl*)> build_topo = [&](ValueImpl* v) {
            if(visited.count(v)) return;

            visited.insert(v);

            for(const auto& p : v->prev) {
                build_topo(p.get());
            }

            topo.push_back(v);
        };

        build_topo(impl.get());

        impl->grad = 1.0;

        for(auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if((*it)->backward) (*it)->backward();
        }

    }

    friend std::ostream& operator<<(std::ostream& os, const Value& v){
        return os << "value(" << v.get() << ")";
    }

    // Addition
    Value operator+(const Value& other) const {
        auto value = std::make_shared<ValueImpl>();
        value->data = impl->data + other.impl->data;
        value->op = "+";
        value->prev = {impl, other.impl};

        /*
            c = a + b;
            dc/da = 1;
            dc/db = 1;

            if f is a function of c, from chain rule
            df/da  = df/dc * dc/da = df/dc * 1;
        */

        auto a_impl = impl;
        auto b_impl = other.impl;
        ValueImpl* out_raw = value.get();

        out_raw->backward = [a_impl, b_impl, out_raw] () {
            a_impl->grad += out_raw->grad;
            b_impl->grad += out_raw->grad;
        };

        return Value{value};
    }

    Value operator+(double val) const {
        return *this + Value{val};
    }

    friend Value operator+(double val, const Value& rhs) {
        return Value{val} + rhs;
    }

    // Subtraction
    Value operator-(const Value& other) const {
        auto value = std::make_shared<ValueImpl>();
        value->data = impl->data - other.impl->data;
        value->op = "-";
        value->prev = {impl, other.impl};

        /*
            c = a - b;
            dc/da = 1;
            dc/db = -1;

            if f is a function of c, from chain rule
            df/da  = df/dc * dc/da = df/dc * 1;
        */

        auto a_impl = impl;
        auto b_impl = other.impl;
        ValueImpl* out_raw = value.get();

        out_raw->backward = [a_impl, b_impl, out_raw] () {
            a_impl->grad += out_raw->grad;
            b_impl->grad += -out_raw->grad;
        };
        return Value{value};
    }

    Value operator-(double val) const {
        return *this - Value{val};
    }

    friend Value operator-(double val, const Value& rhs) {
        return Value{val} - rhs;
    }

    // Multiplication
    Value operator*(const Value& other) const {
        auto value = std::make_shared<ValueImpl>();
        value->data = impl->data * other.impl->data;
        value->op = "*";
        value->prev = {impl, other.impl};

        /*
            c = a * b;
            dc/da = b;
            dc/db = a;

            if f is a function of c, from chain rule
            df/da  = df/dc * dc/da = df/dc * b;
        */

        auto a_impl = impl;
        auto b_impl = other.impl;
        ValueImpl* out_raw = value.get();

        out_raw->backward = [a_impl, b_impl, out_raw] () {
            a_impl->grad += out_raw->grad * b_impl->data;
            b_impl->grad += out_raw->grad * a_impl->data;
        };
        return Value{value};
    }

    Value operator*(double val) const {
        return *this * Value{val};
    }

    friend Value operator*(double val, const Value& rhs) {
        return Value{val} * rhs;
    }

    //Division operator 
    Value operator/(const Value& other) const {
        auto value = std::make_shared<ValueImpl>();
        value->data = impl->data/other.impl->data;
        value->op = "/";
        value->prev = {impl, other.impl};

        /*
            c = a/b;
            dc/da = 1/b;
            dc/db = -a/b^2;

            if f is a function of c, from chain rule
            df/da = df/dc * dc/da = df/dc * 1/b;
            df/db = df/dc * dc/db = df/dc * -a/b^2;
        */

        auto a_impl = impl;
        auto b_impl = other.impl;
        ValueImpl* out_raw = value.get(); 

        value->backward = [a_impl, b_impl, out_raw ]() {
            auto inv_b = (1/b_impl->data);
            a_impl->grad += out_raw->grad * inv_b;
            b_impl->grad += out_raw->grad * -a_impl->data * inv_b * inv_b;
        };

        return Value{value};
    }

    Value operator/(double val) const {
        return *this/Value{val};
    }

    friend Value operator/(double val, const Value& rhs) {
        return Value{val}/rhs;
    }

    //Unary Minus
    Value operator-() const {
        auto value = std::make_shared<ValueImpl>();
        value->data = -impl->data;
        value->op = "neg";
        value->prev = {impl};

        auto a_impl = impl;
        ValueImpl* out_raw = value.get();

        value->backward = [a_impl, out_raw](){
            a_impl->grad += -out_raw->grad;
        };
        
        return Value{value};
    }

    //tanh: tanh(x) = (e^x - e^-x)/(e^x + e^-x)
    friend Value tanh(const Value& a){
        auto value = std::make_shared<ValueImpl>();
        value->data = std::tanh(a.impl->data);
        value->op = "tanh";
        value->prev = {a.impl};

        /*
            b = tanh(a);
            db/da = 1- (tanh(a)^2) = 1 - b^2;
        */

        auto a_impl = a.impl;
        ValueImpl* out_raw = value.get();

        value->backward = [a_impl, out_raw]() {
            a_impl->grad += out_raw->grad * (1 - out_raw->data * out_raw->data);
        };

        return Value{value};

    }

    //relu: relu(x) = max(0, x);
    friend Value relu(const Value& a) {
        auto value = std::make_shared<ValueImpl>();
        value->data = a.impl->data > 0 ? a.impl->data : 0.0;
        value->op = "relu";
        value->prev = {a.impl};

        /*
            b = relu(a);
            db/da = 1 if a > 0.0 else 0.0
            
            c is a function of b, from chain rule
            dc/da = dc/db * db/da = dc/db if a > 0, else 0
        */

        auto a_impl = a.impl;
        ValueImpl* out_raw = value.get();
        value->backward = [a_impl, out_raw]() {
            a_impl->grad += (a_impl->data > 0 ? out_raw->grad : 0.0);
        };

        return Value{value};
    }

    // sigmoid: sigmoid(x) = 1/(1 + e^(-x))
    friend Value sigmoid(const Value& a) {
        auto value = std::make_shared<ValueImpl>();
        value->data = 1.0/(1.0 + std::exp(-a.impl->data));
        value->op = "sigmoid";
        value->prev = {a.impl};

        /*
            b = sigmoid(a);
            db/da = sigmoid(a) * (1 - sigmoid(a));

            dc/da = dc/db * db/da;
        */

        auto a_impl = a.impl;
        ValueImpl* out_raw = value.get();
        out_raw->backward = [a_impl, out_raw]() {
            a_impl->grad += out_raw->grad * (out_raw->data * (1.0 - out_raw->data));
        };

        return Value{value};
    }

    //exp
    friend Value exp(const Value& a) {
        auto value = std::make_shared<ValueImpl>();
        value->data = std::exp(a.impl->data);
        value->op = "exp";
        value->prev = {a.impl};

        /*
            b = exp(a);
            db/da = exp(a);
        */

        auto a_impl = a.impl;
        ValueImpl* out_raw = value.get();
        out_raw->backward = [a_impl, out_raw]() {
            a_impl->grad += out_raw->grad * out_raw->data;
        };

        return Value{value};
    }

    //log
    friend Value log(const Value& a) {
        auto value = std::make_shared<ValueImpl>();
        value->data = std::log(a.impl->data);
        value->op = "log";
        value->prev = {a.impl};

        /*
            b = log(a);
            db/da = 1/a;
        */

        auto a_impl = a.impl;
        ValueImpl* out_raw = value.get();
        out_raw->backward = [a_impl, out_raw]() {
            a_impl->grad += out_raw->grad * (1.0/a_impl->data);
        };

        return Value{value};
    }

    //pow(x, n)
    friend Value pow(const Value& a, double n) {
        auto value = std::make_shared<ValueImpl>();
        value->data = std::pow(a.impl->data, n);
        value->op = "pow";
        value->prev = {a.impl};

        /*
            b = pow(a, n);
            db/da = n * pow(a, n-1);
        */

        auto a_impl = a.impl;
        ValueImpl* out_raw = value.get();
        out_raw->backward = [a_impl, out_raw, n]() {
            a_impl->grad += out_raw->grad * (n * std::pow(a_impl->data, n - 1.0));
        };

        return Value{value};
    }



    double get() const {return impl->data;}
    double grad() const {return impl->grad;}


    std::string to_dot() const {
        std::ostringstream os;
        os << "digraph G {\n";
        os << "  rankdir=LR;\n";
        impl->to_dot_impl(os);
        os << "}\n";
        return os.str();
    }

private:
    std::shared_ptr<ValueImpl> impl;

    explicit Value(std::shared_ptr<ValueImpl> impl) : impl(std::move(impl)){}

};

inline bool gradcheck(std::function<Value(Value)> f, double x_val, double h = 1e-5, double tol = 1e-4) {
    
    // Analytical: build DAG, run backward, read x.grad()
    Value x{x_val};
    Value y = f(x);
    y.backward();
    double analytical = x.grad();

    // Numerical: central differences (no backward needed)
    Value y_plus = f(x_val+h);
    Value y_minus = f(x_val-h);
    double numerical = (y_plus.get() - y_minus.get())/ (2.0 * h);

    double err = std::abs(numerical-analytical);
    bool ok = err < tol;

    std::cout << "grad check analytical = " << analytical << " numerical = " << numerical << " err = " << err << " ok = " << (ok ? "YES" : "NO") << std::endl;

    return ok;

}

