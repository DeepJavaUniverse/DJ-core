package com.kovalevskyi.java.deep.model;

import java.util.Set;

public class InputNeuron implements Neuron {

    private Set<Neuron> connections;

    private Double value;

    @Override
    public Double call() throws Exception {
        return value;
    }

    public void setValue(final Double value) {
        this.value = value;
    }

    @Override
    public void invalidate() { } // No-op

    @Override
    public void addForwardConnection(final Neuron neuron) {
        connections.add(neuron);
    }

    @Override
    public void addBackwardConnection(final Neuron neuron, final Double weight) {
        throw new RuntimeException("addBackwardConnection should be never called on an InputNeuron");
    }
}
