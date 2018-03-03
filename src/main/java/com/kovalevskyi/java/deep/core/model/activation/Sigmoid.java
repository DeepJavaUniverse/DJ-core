package com.kovalevskyi.java.deep.core.model.activation;


public class Sigmoid implements ActivationFunction {

    public Double forward(Double x) {
        return 1. / (1. + Math.exp(- x));
    }

    @Override
    public Double backward(final Double error) {
        return forward(error) * (1 - forward(error));
    }
}
