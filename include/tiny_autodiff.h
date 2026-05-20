#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <functional>
#include <unordered_set>

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

    friend Value operator+(double val, const Value& rhs) {
        return Value{val} + rhs;
}

    friend Value operator-(double val, const Value& rhs) {
        return Value{val} - rhs;
    }

    friend Value operator*(double val, const Value& rhs) {
        return Value{val} * rhs;
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

