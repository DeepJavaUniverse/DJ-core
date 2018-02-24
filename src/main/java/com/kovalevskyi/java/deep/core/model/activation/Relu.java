package com.kovalevskyi.java.deep.core.model.activation;


public class Relu implements ActivationFunction {

    @Override
    public Double forward(final Double x) {
        return x > 0 ? x : 0;
    }

    @Override
    public Double backward(final Double error) {
        return error > 0. ? 1. : 0.;
    }
}
