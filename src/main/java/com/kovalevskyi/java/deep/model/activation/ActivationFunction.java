package com.kovalevskyi.java.deep.model.activation;


public interface ActivationFunction {

    Double forward(final Double x);

    Double backward(final Double error);
}
