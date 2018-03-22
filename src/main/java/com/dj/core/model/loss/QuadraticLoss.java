package com.dj.core.model.loss;


public class QuadraticLoss implements Loss {
    
    @Override
    public double error(final double actual, final double expected) {
        return Math.pow(expected - actual, 2);
    }

    @Override
    public double derivative(final double actual, final double expected) {
        return 2 * (expected - actual);
    }
}
