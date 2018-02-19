package com.kovalevskyi.java.deep.model.activation;

import java.util.function.Function;

public class Sigmoid implements ActivationFunction {

    public Double forward(final Double aDouble) {
        return 1. / (1. + Math.exp(- aDouble));
    }

    @Override
    public synchronized void invalidate() {
        // noop
    }

    @Override
    public Double backward(final Double x) {
        return forward(x) * (1 - forward(x));
    }
}
