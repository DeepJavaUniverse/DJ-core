package com.dj.core.model.graph;

import java.util.HashSet;
import java.util.Set;

public class InputNeuron implements Neuron {

    private Set<Neuron> connections = new HashSet<>();

    private final String name;

    public InputNeuron(final String name) {
        this.name = name;
    }

    public InputNeuron() {
        this(null);
    }

    @Override
    public void forwardSignalReceived(final Neuron from, final Double... values) {
        connections.forEach(n -> n.forwardSignalReceived(this, values));
    }

    @Override
    public void backwardSignalReceived(final Double value) {
        // No-op
    }

    @Override
    public void addForwardConnection(final Neuron neuron) {
        connections.add(neuron);
    }

    @Override
    public void addBackwardConnection(final Neuron neuron, final Double weight) {
        throw new RuntimeException("addBackwardConnection should be never called on an InputNeuron");
    }

    @Override
    public double[] getForwardResult() {
        return new double[] { 0 };
    }

    @Override
    public String toString() {
        if (name != null) {
            return name;
        }
        return super.toString();
    }
}
