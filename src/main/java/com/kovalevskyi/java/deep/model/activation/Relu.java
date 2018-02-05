package com.kovalevskyi.java.deep.model.activation;

import java.util.function.Function;

public class Relu implements ActivationFunction {

    @Override
    public Double forward(final Double aDouble) {
        return aDouble > 0 ? aDouble : 0;
    }

    @Override
    public Double backward(final Double x) {
        return x > 0. ? 1. : 0.;
    }
}
