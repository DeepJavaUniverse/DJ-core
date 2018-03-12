package com.kovalevskyi.java.deep.core.model.activation;


public class Sigmoid implements ActivationFunction {

    private final boolean enforceLimit;

    public Sigmoid(final boolean enforceLimit) {
        this.enforceLimit = enforceLimit;
    }

    public Sigmoid() {
        this(false);
    }

    public Double forward(Double x) {
        return 1. / (1. + Math.exp(- x));
    }

    @Override
    public Double backward(final Double error) {
        if (enforceLimit && error > 1.) {
            return forward(1.) * (1 - forward(1.));
        } else if (enforceLimit && error < -1.) {
            return forward(-1.) * (1 - forward(-1.));
        }
        return forward(error) * (1 - forward(error));
    }
}
