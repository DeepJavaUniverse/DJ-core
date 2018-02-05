package com.kovalevskyi.java.deep.model.activation;

import jdk.jshell.spi.ExecutionControl;

public interface ActivationFunction {

    public Double forward(final Double x);

    public default Double backward(final Double x) {
        throw new RuntimeException("backward is not yet implemented");
    }

}
