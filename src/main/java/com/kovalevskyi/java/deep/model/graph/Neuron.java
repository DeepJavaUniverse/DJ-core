package com.kovalevskyi.java.deep.model.graph;

import java.util.Set;

public interface Neuron {

    void forwardInvalidate();

    Set<Neuron> backwardConnections();

    Boolean forwardCalculated();

    Double forwardResult();

    Boolean canBeCalculatedForward();

    Double calculateForward();

    void addForwardConnection(Neuron neuron);

    void addBackwardConnection(Neuron neuron, Double weight);

}
