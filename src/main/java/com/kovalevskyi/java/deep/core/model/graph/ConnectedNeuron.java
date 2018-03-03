package com.kovalevskyi.java.deep.core.model.graph;

import com.google.common.util.concurrent.AtomicDouble;
import com.kovalevskyi.java.deep.core.model.activation.ActivationFunction;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class ConnectedNeuron implements Neuron {

    private final ActivationFunction activationFunction;

    private final Map<Neuron, Double> backwardConnections = new HashMap<>();

    private final Set<Neuron> forwardConnections = new HashSet<>();

    private final double learningRate;

    private final String name;

    private final Map<Neuron, Double> inputSignals = new HashMap<>();

    private volatile int signalReceived;

    private final AtomicDouble bias;

    private volatile boolean forwardCalculated;

    private volatile double forwardResult;

    private volatile double inputSignalsAverage;

    private volatile double forwardInputToActivationFunction;

    private ConnectedNeuron(
            final ActivationFunction activationFunction,
            final double learningRate,
            final String name,
            final double bias) {
        this.activationFunction = activationFunction;
        this.learningRate = learningRate;
        this.name = name;
        this.bias = new AtomicDouble(bias);
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
            signalReceived = 0;
        }
    }

    @Override
    public void forwardSignalReceived(final Neuron from, final Double value) {
        signalReceived++;
        if (!inputSignals.containsKey(from)) {
            throw new RuntimeException(
                    String.format(
                            "Neuron %s is not connected to the %s",
                            from,
                            this));
        }
        inputSignals.put(from, value);
        inputSignalsAverage += inputSignalsAverage;
        if (backwardConnections.size() == signalReceived) {
            forwardInputToActivationFunction
                    = backwardConnections
                        .entrySet()
                        .stream()
                        .mapToDouble(connection ->
                                inputSignals.get(connection.getKey())
                                        * connection.getValue())
                        .sum() + bias.get();
            double signalToSend
                    = activationFunction.forward(
                            forwardInputToActivationFunction);
            forwardResult = signalToSend;
            forwardCalculated = true;

            forwardConnections
                    .stream()
                    .forEach(connection -> {
                        connection
                                .forwardSignalReceived(
                                        ConnectedNeuron.this,
                                        signalToSend);
                    });
            inputSignalsAverage
                    = inputSignalsAverage / (double) signalReceived;
            signalReceived = 0;
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
        bias.addAndGet(inputSignalsAverage * dz * learningRate);
        backwardConnections
                .keySet()
                .stream()
                .forEach(conn ->
                    conn.backwardSignalReceived(backwardConnections.get(conn) * dz)
                );
    }

    @Override
    public void addForwardConnection(final Neuron neuron) {
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

    public static class Builder {

        private double bias = new Random().nextDouble();

        private String name;

        private ActivationFunction activationFunction;

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

        public ConnectedNeuron build() {
            if (activationFunction == null) {
                throw new RuntimeException("ActivationFunction need to be set in order to" +
                        " create a ConnectedNeuron");
            }
            return new ConnectedNeuron(
                    activationFunction,
                    learningRate,
                    name,
                    bias);
        }
    }
}
