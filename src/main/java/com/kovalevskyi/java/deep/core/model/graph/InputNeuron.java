package com.kovalevskyi.java.deep.core.model.graph;

import java.util.HashSet;
import java.util.Set;

public class InputNeuron implements Neuron {

    private Set<Neuron> connections = new HashSet<>();

    @Override
    public void forwardSignalReceived(final Neuron from, final Double value) {
        connections.forEach(n -> n.forwardSignalReceived(this, value));
    }

    @Override
    public void addForwardConnection(final Neuron neuron) {
        connections.add(neuron);
    }

    @Override
    public void addBackwardConnection(final Neuron neuron, final Double weight) { }
}
