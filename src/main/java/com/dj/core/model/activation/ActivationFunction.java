package com.dj.core.model.activation;


import java.io.Serializable;

public interface ActivationFunction extends Serializable {

    Double forward(final Double x);

    Double backward(final Double error);
}
