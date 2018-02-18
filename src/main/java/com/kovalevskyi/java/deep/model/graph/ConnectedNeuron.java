package com.kovalevskyi.java.deep.model.graph;

import com.kovalevskyi.java.deep.model.activation.ActivationFunction;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class ConnectedNeuron implements Neuron {

    private final Map<Neuron, Double> backwardConnections;

    private final Set<Neuron> forwardConnections = new HashSet<>();

    private final double bias;

    private final ActivationFunction activationFunction;

    private volatile double forwardResult;

    private volatile boolean forwardCalculated;

    public ConnectedNeuron(final Map<Neuron, Double> backwardConnections,
                           final double bias,
                           final ActivationFunction activationFunction) {
        
        this.backwardConnections = backwardConnections;
        this.bias = bias;
        this.activationFunction = activationFunction;
    }


    @Override
    public void forwardInvalidate() {
        if (forwardCalculated) {
            forwardCalculated = false;
            backwardConnections.keySet().forEach(Neuron::forwardInvalidate);
        }
    }

    @Override
    public Set<Neuron> backwardConnections() {
        return backwardConnections.keySet();
    }

    @Override
    public Boolean forwardCalculated() {
        return forwardCalculated;
    }

    @Override
    public Double forwardResult() {
        if (!forwardCalculated) {
            throw new RuntimeException("Not yet forwardCalculated");
        }
        return forwardResult;
    }

    @Override
    public Boolean canBeCalculatedForward() {
        return !backwardConnections.keySet().stream().filter(n -> !n.forwardCalculated()).findAny().isPresent();
    }

    @Override
    public Double calculateForward() {
        if (forwardCalculated) {
            return forwardResult;
        }
        if (backwardConnections
                .keySet()
                .stream()
                .map(Neuron::forwardCalculated)
                .<Boolean, Boolean>filter(calculated -> !calculated)
                .count() > 0) {
           throw new RuntimeException("Not all neurons are forwardCalculated!");
        }
        forwardResult = backwardConnections
                .entrySet()
                .stream()
                .mapToDouble(connection -> connection.getKey().forwardResult() * connection.getValue())
                .sum() + bias;
        forwardResult = activationFunction.forward(forwardResult);
        forwardCalculated = true;
        return forwardResult;
    }

    @Override
    public void addForwardConnection(final Neuron neuron) {
        forwardConnections.add(neuron);
    }

    @Override
    public void addBackwardConnection(final Neuron neuron, final Double weight) {
        backwardConnections.put(neuron, weight);
    }
}
