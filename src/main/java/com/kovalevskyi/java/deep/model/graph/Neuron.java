package com.kovalevskyi.java.deep.model.graph;

import java.util.Map;
import java.util.Set;

public interface Neuron {

    void forwardInvalidate();

    Set<Neuron> backwardConnections();

    void forwardSignalReceived(Neuron from, Double value);

    void backwardSignalReceived(Double value);

    void addForwardConnection(Neuron neuron);

    void addBackwardConnection(Neuron neuron, Double weight);

    Map<Neuron, Double> getBackwardConnections();

    double getBias();

}
