package com.kovalevskyi.java.deep.core.model.activation;

public class HyperbolicTangent implements ActivationFunction {
    @Override
    public Double forward(final Double x) {
        return Math.sinh(x) / Math.cosh(x);
    }

    @Override
    public Double backward(final Double error) {
        final double tanhError = error;
        return 1. - tanhError * tanhError;
    }
}
