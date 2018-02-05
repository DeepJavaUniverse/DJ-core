package com.kovalevskyi.java.deep.model;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Function;

public class ConnectedNeuron implements Neuron {

    private static final ForkJoinPool FORK_JOIN_POOL = new ForkJoinPool();

    private final Map<Neuron, Double> backwardConnections;

    private final Set<Neuron> forwardConnections = new HashSet<>();

    private final double bias;

    private final Function<Double, Double> activationFunction;

    private volatile double result;

    private volatile boolean calculated;

    public ConnectedNeuron(final Map<Neuron, Double> backwardConnections,
                           final double bias, final Function<Double, Double> activationFunction) {
        
        this.backwardConnections = backwardConnections;
        this.bias = bias;
        this.activationFunction = activationFunction;
    }

    @Override
    public Double call() throws Exception {
        if (calculated){
            return result;
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
                .sum();
        result = activationFunction.apply(connectionsProcessingResult + bias);
        calculated = true;
        return result;
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
