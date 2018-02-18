package com.kovalevskyi.java.deep.model.graph;

import com.kovalevskyi.java.deep.model.activation.ActivationFunction;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class ConnectedNeuron implements Neuron {

    private static final ForkJoinPool FORK_JOIN_POOL = new ForkJoinPool();

    private final Map<Neuron, Double> backwardConnections;

    private final Set<Neuron> forwardConnections = new HashSet<>();

    private final double bias;

    private final ActivationFunction activationFunction;

    private volatile Map<Neuron, Double> inputSignals = new ConcurrentHashMap<>();

    private volatile boolean forwardCalculated;

    private volatile double forwardResult;

    public ConnectedNeuron(final Map<Neuron, Double> backwardConnections,
                           final double bias,
                           final ActivationFunction activationFunction) {
        
        this.backwardConnections = backwardConnections;
        backwardConnections.keySet().forEach(n -> inputSignals.put(n, Double.NaN));
        this.bias = bias;
        this.activationFunction = activationFunction;
    }

    public double getForwardResult() {
        return forwardResult;
    }

    @Override
    public void forwardInvalidate() {
        if (forwardCalculated) {
            forwardCalculated = false;
            inputSignals.forEach((in, v) -> inputSignals.put(in, Double.NaN));
            backwardConnections.keySet().forEach(Neuron::forwardInvalidate);
        }
    }

    @Override
    public Set<Neuron> backwardConnections() {
        return backwardConnections.keySet();
    }

    @Override
    public void forwardSignalReceived(final Neuron from, final Double value) {
        FORK_JOIN_POOL.submit(new RecursiveAction() {
            @Override
            protected void compute() {
                inputSignals.put(from, value);
                Long notNullCount = inputSignals.values().stream().filter(v -> v != Double.NaN).count();
                if (notNullCount == inputSignals.keySet().size()) {
                    double signalToSend = activationFunction.forward(backwardConnections
                            .entrySet()
                            .stream()
                            .mapToDouble(connection -> inputSignals.get(connection.getKey()) * connection.getValue())
                            .sum() + bias);
                    forwardResult = signalToSend;
                    forwardCalculated = true;
                    forwardConnections.forEach(c -> c.forwardSignalReceived(ConnectedNeuron.this, signalToSend));
                }
            }
        });
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
