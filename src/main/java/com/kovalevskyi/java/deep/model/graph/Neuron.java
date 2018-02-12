package com.kovalevskyi.java.deep.model.graph;

import java.util.Set;

public interface Neuron {

    void invalidate();

    Set<Neuron> backwardConnections();

    Boolean calculated();

    Double forwardResult();

    Boolean canBeCalculatedForward();

    Double calculateForward();

    void addForwardConnection(Neuron neuron);

    void addBackwardConnection(Neuron neuron, Double weight);

}
