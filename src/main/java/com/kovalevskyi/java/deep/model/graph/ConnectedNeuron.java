package com.kovalevskyi.java.deep.model.graph;

import com.kovalevskyi.java.deep.model.activation.ActivationFunction;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ConnectedNeuron implements Neuron {

    private final ForkJoinPool forkJoinPool;

    private final Map<Neuron, Double> backwardConnections;

    private final Set<Neuron> forwardConnections = new HashSet<>();

    private double bias;

    private final ActivationFunction activationFunction;

    private volatile Map<Neuron, Double> inputSignals = new ConcurrentHashMap<>();

    private volatile boolean forwardCalculated;

    private volatile double forwardResult;

    private final double learningRate;

    private final String name;

    public ConnectedNeuron(final Map<Neuron, Double> backwardConnections,
                           final double bias,
                           final ActivationFunction activationFunction,
                           final double learningRate,
                           final ForkJoinPool forkJoinPool) {
        this(backwardConnections,
                bias,
                activationFunction,
                learningRate,
                null,
                forkJoinPool);
    }

    public ConnectedNeuron(final Map<Neuron, Double> backwardConnections,
                           final double bias,
                           final ActivationFunction activationFunction,
                           final double learningRate,
                           final String name,
                           final ForkJoinPool forkJoinPool) {
        this.forkJoinPool = forkJoinPool;
        this.backwardConnections = new ConcurrentHashMap<>(backwardConnections);
        this.learningRate = learningRate;
        backwardConnections.keySet().forEach(n -> inputSignals.put(n, Double.NaN));
        this.bias = bias;
        this.activationFunction = activationFunction;
        this.name = name;
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
            forwardConnections.stream().map(c ->
                new RecursiveAction() {
                    @Override
                    protected void compute() {
                      c.forwardSignalReceived(ConnectedNeuron.this, signalToSend);
                    }
                }
            ).map(action -> {
                forkJoinPool.submit(action);
                return action;
            }).forEach(RecursiveAction::join);
        }
    }

    @Override
    public void backwardSignalReceived(final Double error) {
        if (!forwardCalculated) {
            throw new RuntimeException("Forward calculation is not yet completed");
        }
        double backwardDiff = activationFunction.backward(forwardResult);
        double dz = backwardDiff * error;
        backwardConnections.keySet().forEach(conn -> {
            double weight = backwardConnections.get(conn);
            weight = weight + inputSignals.get(conn) * dz * learningRate;
            backwardConnections.put(conn, weight);
        });
        double average = inputSignals.values().stream().mapToDouble(w -> w).average().getAsDouble();
        bias = bias + average * dz * learningRate;
        backwardConnections.keySet().stream().map(conn ->
            new RecursiveAction() {
                @Override
                protected void compute() {
                    conn.backwardSignalReceived(backwardConnections.get(conn) * dz);
                }
            }
        ).map(action -> {
            forkJoinPool.submit(action);
            return action;
        }).forEach(RecursiveAction::join);
    }

    @Override
    public void addForwardConnection(final Neuron neuron) {
        forwardConnections.add(neuron);
    }

    @Override
    public void addBackwardConnection(final Neuron neuron, final Double weight) {
        backwardConnections.put(neuron, weight);
    }

    @Override
    public Map<Neuron, Double> getBackwardConnections() {
        return backwardConnections;
    }

    @Override
    public double getBias() {
        return bias;
    }

    @Override
    public String toString() {
        if (name != null) {
            return name;
        }
        return super.toString();
    }
}
