package com.dj.core.model.activation;


public interface ActivationFunction {

    Double forward(final Double x);

    Double backward(final Double error);
}
