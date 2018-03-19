package com.kovalevskyi.java.deep.core.model.graph;

import com.google.common.util.concurrent.AtomicDouble;
import com.kovalevskyi.java.deep.core.model.activation.ActivationFunction;

import java.util.*;

public class ConnectedNeuron implements Neuron {

    private final ActivationFunction activationFunction;

    private final Map<Neuron, Double> backwardConnections = new HashMap<>();

    private final Set<Neuron> forwardConnections = new HashSet<>();

    private final double learningRate;

    private final String name;

    private final Map<Neuron, Double> inputSignals = new HashMap<>();

    private final AtomicDouble bias;

    private final boolean debug;

    private volatile int signalReceived;

    private volatile double forwardResult;

    private volatile double inputSignalsAverageTmp;

    private volatile double inputSignalsAverage;

    private volatile double forwardInputToActivationFunction;

    private ConnectedNeuron(
            final ActivationFunction activationFunction,
            final double learningRate,
            final String name,
            final double bias,
            final boolean debug) {
        this.activationFunction = activationFunction;
        this.learningRate = learningRate;
        this.name = name;
        this.bias = new AtomicDouble(bias);
        this.debug = debug;
    }

    @Override
    public double getForwardResult() {
        return forwardResult;
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
        inputSignalsAverageTmp += value;
        if (backwardConnections.size() == signalReceived) {
            forwardInputToActivationFunction
                    = backwardConnections
                        .entrySet()
                        .stream()
                        .mapToDouble(connection ->
                                inputSignals.get(connection.getKey())
                                        * connection.getValue())
                        .sum() + bias.get();
            if (debug) {
                if (brokenValue(forwardInputToActivationFunction)) {
                    throw new RuntimeException("Forward input to activation function is broken");
                }
            }
            double signalToSend
                    = activationFunction.forward(
                            forwardInputToActivationFunction);
            forwardResult = signalToSend;

            forwardConnections
                    .stream()
                    .forEach(connection -> {
                        connection
                                .forwardSignalReceived(
                                        ConnectedNeuron.this,
                                        signalToSend);
                    });
            inputSignalsAverage
                    = inputSignalsAverageTmp / (double) signalReceived;
            inputSignalsAverageTmp = 0.;
            signalReceived = 0;
        }
    }

    @Override
    public void backwardSignalReceived(final Double error) {
        if (!forwardCalculated()) {
            throw new RuntimeException("Forward calculation is not yet completed");
        }
        final double derivative
                = activationFunction.backward(
                        forwardInputToActivationFunction);

        double dz = derivative * error;
        if (debug) {
            if (brokenValue(derivative)) {
                throw new RuntimeException("derivative value is broken");
            } else if (brokenValue(dz) || (error != 0. && dz == 0.)) {
                throw new RuntimeException("dz value is broken");
            } else if (brokenValue(error)) {
                throw new RuntimeException("error value is broken");
            }
        }
        if (error == 0.) {
            return;
        }
        if (derivative == 0.) {
            return;
        }
        backwardConnections.keySet().forEach(conn ->
            backwardConnections.compute(conn, (k, weight) -> {
                    final double newWeight = weight + inputSignals.get(conn) * dz * learningRate;
                    if (debug) {
                        if (brokenValue(newWeight)) {
                            throw new RuntimeException("Updated connection weight is broken");
                        }
                    }
                    return newWeight;
                }
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

    private static boolean brokenValue(final double value) {
        if (value == Double.MAX_VALUE) {
            return true;
        }
        if (value == Double.MIN_VALUE) {
            return true;
        }
        if (value == Double.NaN) {
            return true;
        }
        if (value == Double.NEGATIVE_INFINITY) {
            return true;
        }
        if (value == Double.POSITIVE_INFINITY) {
            return true;
        }
        return false;
    }

    private boolean forwardCalculated() {
        return signalReceived == 0;
    }

    public static class Builder {

        private double bias = new Random().nextDouble();

        private String name;

        private ActivationFunction activationFunction;

        private double learningRate = 0.2;

        private boolean debug;

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

        public Builder debug(final boolean debug) {
            this.debug = debug;
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
                    bias,
                    debug);
        }
    }
}
