package com.kovalevskyi.java.deep.model.graph;

import java.util.Set;

public class InputNeuron implements Neuron {

    private Set<Neuron> connections;

    private Double value;

    public void setValue(final Double value) {
        this.value = value;
    }

    @Override
    public void invalidate() { } // No-op

    @Override
    public Set<Neuron> backwardConnections() {
        return Set.of();
    }

    @Override
    public Boolean calculated() {
        return true;
    }

    @Override
    public Double forwardResult() {
        return value;
    }

    @Override
    public Boolean canBeCalculatedForward() {
        return true;
    }

    @Override
    public Double calculateForward() {
        return value;
    }

    @Override
    public void addForwardConnection(final Neuron neuron) {
        connections.add(neuron);
    }

    @Override
    public void addBackwardConnection(final Neuron neuron, final Double weight) {
        throw new RuntimeException("addBackwardConnection should be never called on an InputNeuron");
    }
}
