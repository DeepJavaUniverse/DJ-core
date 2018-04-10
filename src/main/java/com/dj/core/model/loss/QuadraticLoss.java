package com.dj.core.model.loss;


import java.util.stream.IntStream;

public class QuadraticLoss implements Loss {
    
    @Override
    public double error(final double actual, final double expected) {
        return Math.pow(expected - actual, 2);
    }

    @Override
    public double[] derivative(final double[] actual, final double[] expected) {
        double[] res = new double[actual.length];
        IntStream.range(0, actual.length).forEach(i -> {
            res[i] = 2 * (expected[i] - actual[i]);
        });
        return res;
    }
}
