package com.kovalevskyi.java.deep.core.model.activation;


public class StepFunction implements ActivationFunction {

    public Double forward(Double x) {
        return x >= 0.5 ? 1. : 0.;
    }
}
