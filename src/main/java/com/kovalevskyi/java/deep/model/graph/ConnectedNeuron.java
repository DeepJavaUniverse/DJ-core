package com.kovalevskyi.java.deep.model.graph;

import com.kovalevskyi.java.deep.model.activation.ActivationFunction;

import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;
import java.util.function.Function;

public class ConnectedNeuron extends Neuron {

    private final ActivationFunction activationFunction;

    private final ForkJoinPool forkJoinPool;

    private final ConcurrentHashMap<Neuron, Double> backwardConnections
            = new ConcurrentHashMap<>();

    private final Set<Neuron> forwardConnections = new HashSet<>();

    private final double learningRate;

    private final String name;

    private final Map<Neuron, Double> inputSignals = new ConcurrentHashMap<>();

    private volatile double bias;

    private volatile boolean forwardCalculated;

    private volatile double forwardResult;

    private volatile double inputSignalsAverage;

    private volatile double forwardInputToActivationFunction;

    private ConnectedNeuron(final ConnectedNeuron.Builder builder) {
        this.activationFunction = builder.activationFunction;
        this.forkJoinPool = builder.forkJoinPool;
        this.learningRate = builder.learningRate;
        this.name = builder.name;
        this.bias = builder.bias;
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
    public void forwardSignalReceived(final Neuron from, final Double value) {
        if (!inputSignals.containsKey(from)) {
            throw new RuntimeException(
                    String.format(
                            "Neuron %s is not connected to the %s",
                            from,
                            this));
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
    public void connect(final Neuron neuron, final Double weight) {
        neuron.addBackwardConnection(this, weight);
        forwardConnections.add(neuron);
    }

    @Override
    public void addBackwardConnection(final Neuron neuron, final Double weight) {
        backwardConnections.put(neuron, weight);
        inputSignals.put(neuron, Double.NaN);
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

    public static class Builder {

        private double bias = new Random().nextDouble();

        private String name;

        private ActivationFunction activationFunction;

        private ForkJoinPool forkJoinPool;

        private double learningRate = 0.2;

        public Builder bias(final double bias) {
            this.bias = bias;
            return this;
        }

        public Builder name(final String name) {
            this.name = name;
            return this;
        }

        public Builder activationFunction(final ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public Builder learningRate(final double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder forkJoinPool(final ForkJoinPool forkJoinPool) {
            this.forkJoinPool = forkJoinPool;
            return this;
        }

        public ConnectedNeuron build() {
            if (forkJoinPool == null) {
                throw new RuntimeException("ForkJoinPool need to be set in order to" +
                        " create a ConnectedNeuron");
            }
            if (activationFunction == null) {
                throw new RuntimeException("ActivationFunction need to be set in order to" +
                        " create a ConnectedNeuron");
            }
            return new ConnectedNeuron(this);
        }
    }
}
