package com.dj.core.model.loss;


import java.util.stream.IntStream;

public class QuadraticLoss implements Loss {
    
    @Override
    public double error(final double[] actual, final double[] expected) {
        return IntStream
                .range(0, actual.length)
                .mapToDouble(i -> Math.pow(expected[i] - actual[i], 2))
                .sum();
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
