package com.kovalevskyi.java.deep.model;

import java.util.concurrent.Callable;

public interface Neuron extends Callable<Double> {

    void invalidate();

    void addForwardConnection(Neuron neuron);

    void addBackwardConnection(Neuron neuron, Double weight);

}
