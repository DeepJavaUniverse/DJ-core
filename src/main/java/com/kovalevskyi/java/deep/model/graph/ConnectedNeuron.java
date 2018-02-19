package com.kovalevskyi.java.deep.model.graph;

import com.kovalevskyi.java.deep.model.activation.ActivationFunction;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;
import java.util.function.Function;

public class ConnectedNeuron implements Neuron {

    private final ActivationFunction activationFunction;

    private final ForkJoinPool forkJoinPool;

    private final ConcurrentHashMap<Neuron, Double> backwardConnections;

    private final Set<Neuron> forwardConnections = new HashSet<>();

    private final double learningRate;

    private final String name;

    private final Map<Neuron, Double> inputSignals = new ConcurrentHashMap<>();

    private volatile double bias;

    private volatile boolean forwardCalculated;

    private volatile double forwardResult;

    private volatile double inputSignalsAverage;

    private volatile double forwardInputToActivationFunction;

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
        if (!inputSignals.containsKey(from)) {
            throw new RuntimeException(
                    String.format("Neuron %s is not connected", from));
        }
        inputSignals.put(from, value);
        if (isAllSignalsReceived()) {
            forwardInputToActivationFunction
                    = backwardConnections
                        .entrySet()
                        .stream()
                        .mapToDouble(connection ->
                                inputSignals.get(connection.getKey())
                                        * connection.getValue())
                        .sum() + bias;
            double signalToSend
                    = activationFunction.forward(
                            forwardInputToActivationFunction);
            forwardResult = signalToSend;
            forwardCalculated = true;
            forwardConnections
                    .stream()
                    .map(connection ->
                        new RecursiveAction() {
                            @Override
                            protected void compute() {
                              connection
                                      .forwardSignalReceived(
                                              ConnectedNeuron.this,
                                              signalToSend);
                            }
                        }
                    ).map(forkJoinPool::submit)
                    .forEach(ForkJoinTask::join);
            inputSignalsAverage
                    = inputSignals
                        .values()
                        .stream()
                        .mapToDouble(v -> v)
                        .average()
                        .getAsDouble();
        }
    }

    @Override
    public void backwardSignalReceived(final Double error) {
        if (!forwardCalculated) {
            throw new RuntimeException("Forward calculation is not yet completed");
        }
        double derivative
                = activationFunction.backward(
                        forwardInputToActivationFunction);
        double dz = derivative * error;
        backwardConnections.keySet().forEach(conn ->
            backwardConnections.compute(conn, (k, weight) ->
               weight + inputSignals.get(conn) * dz * learningRate
            ));
        // TODO(issues/9): bias update is not Thread safe.
        bias = bias + inputSignalsAverage * dz * learningRate;
        backwardConnections
                .keySet()
                .stream()
                .map(conn ->
                    new RecursiveAction() {
                        @Override
                        protected void compute() {
                            conn.backwardSignalReceived(backwardConnections.get(conn) * dz);
                        }
                    }
                ).map(forkJoinPool::submit)
                .forEach(ForkJoinTask::join);
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

    private boolean isAllSignalsReceived() {
        Long notNullCount
                = inputSignals
                    .values()
                    .stream()
                    .filter(v -> v != Double.NaN)
                    .count();
        return notNullCount == inputSignals.keySet().size();
    }
}
