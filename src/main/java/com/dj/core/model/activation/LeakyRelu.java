package com.dj.core.model.activation;


public class LeakyRelu implements ActivationFunction {

    private final double m;

    public LeakyRelu(final double m) {
        this.m = m;
    }

    public LeakyRelu() {
        this(0.0001);
    }

    @Override
    public Double forward(final Double x) {
        return x > 0 ? x : m * x;
    }

    @Override
    public Double backward(final Double error) {
        return error > 0. ? 1. : m;
    }
}
