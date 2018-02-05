package com.kovalevskyi.java.deep.model.activation;

import java.util.function.Function;

public class Sigmoid implements Function<Double, Double> {
    @Override
    public Double apply(final Double aDouble) {
        return 1. / (1. + Math.exp(- aDouble));
    }
}
