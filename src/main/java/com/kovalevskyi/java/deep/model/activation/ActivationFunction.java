package com.kovalevskyi.java.deep.model.activation;

import jdk.jshell.spi.ExecutionControl;

public interface ActivationFunction {

    Double forward(final Double x);

    void invalidate();

    default Double backward(final Double x) {
        throw new RuntimeException("backward is not yet implemented");
    }

}
