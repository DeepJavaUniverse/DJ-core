package com.kovalevskyi.java.deep.core.model.activation;


public class Sigmoid implements ActivationFunction {

    public Double forward(Double x) {
        return 1. / (1. + Math.exp(- x));
    }
}
