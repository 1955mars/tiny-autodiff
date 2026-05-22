#pragma once

#include "tiny_autodiff.h"

#include <random>
#include <cassert>

class Neuron {
public:
    explicit Neuron(size_t n_in):bias(Value{0.0}) {
        static std::mt19937 rng{4};
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        weights.reserve(n_in);
        for(size_t i = 0; i<n_in; i++) {
            weights.emplace_back(dist(rng));
        }
    }

    Value operator()(const std::vector<Value>& xs) const {
        assert(xs.size() == weights.size());
        Value sum = bias;
        for(size_t i = 0; i < xs.size(); i++) {
            sum = sum + weights[i] * xs[i]; // each op extends the DAG
        }
        return tanh(sum);
    }

    std::vector<Value> parameters() const {
        auto params = weights;
        params.emplace_back(bias);
        return params;
    }

private:
    std::vector<Value> weights;
    Value bias;
};