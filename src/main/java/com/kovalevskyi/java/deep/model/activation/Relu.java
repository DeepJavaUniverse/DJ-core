package com.kovalevskyi.java.deep.model.activation;

import java.util.function.Function;

public class Relu implements Function<Double, Double> {
    @Override
    public Double apply(final Double aDouble) {
        return aDouble > 0 ? aDouble : 0;
    }
}
