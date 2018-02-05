package com.kovalevskyi.java.deep.model;

import com.kovalevskyi.java.deep.model.activation.ActivationFunction;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;

public class ConnectedNeuron implements Neuron {

    private static final ForkJoinPool FORK_JOIN_POOL = new ForkJoinPool();

    private final Map<Neuron, Double> backwardConnections;

    private final Set<Neuron> forwardConnections = new HashSet<>();

    private final double bias;

    private final ActivationFunction activationFunction;

    private final boolean learning;

    private volatile double forwardResult;

    private volatile double backwardResult;

    private volatile boolean calculated;

    public ConnectedNeuron(final Map<Neuron, Double> backwardConnections,
                           final double bias,
                           final ActivationFunction activationFunction,
                           final boolean learning) {
        
        this.backwardConnections = backwardConnections;
        this.bias = bias;
        this.activationFunction = activationFunction;
        this.learning = learning;
    }

    @Override
    public Double call() throws Exception {
        if (calculated){
            return forwardResult;
        }
        double connectionsProcessingResult = backwardConnections
                .entrySet()
                .stream()
                .map(connection -> Map.entry(FORK_JOIN_POOL.submit(connection.getKey()), connection.getValue()))
                .map(task -> {
                    try {
                        return Map.entry(task.getKey().get(), task.getValue());
                    } catch (Exception e) {
                        throw new RuntimeException("Neuron processing exception", e);
                    }
                }).mapToDouble(res -> res.getKey() * res.getValue())
                .sum() + bias;
        forwardResult = activationFunction.forward(connectionsProcessingResult);
        if (learning) {
             backwardResult = activationFunction.backward(connectionsProcessingResult);
        }
        calculated = true;
        return forwardResult;
    }

    @Override
    public void invalidate() {
        calculated = false;
        backwardConnections.keySet().forEach(Neuron::invalidate);
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
