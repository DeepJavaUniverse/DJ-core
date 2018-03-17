package com.kovalevskyi.java.deep.core.model.graph;

import com.google.common.util.concurrent.AtomicDouble;
import com.kovalevskyi.java.deep.core.model.activation.ActivationFunction;

import java.util.*;

public class ConnectedNeuron implements Neuron {

    private final ActivationFunction activationFunction;
    private final Map<Neuron, Double> backwardConnections = new HashMap<>();
    private final Set<Neuron> forwardConnections = new HashSet<>();
    private final Map<Neuron, Double> inputSignals = new HashMap<>();
    private volatile int signalReceived;
    private final AtomicDouble bias;
    private volatile double forwardResult;
    private volatile double inputSignalsAverage;

    private ConnectedNeuron(
            final ActivationFunction activationFunction,
            final double bias) {
        this.activationFunction = activationFunction;
        this.bias = new AtomicDouble(bias);
    }

    public double getForwardResult() {
        return forwardResult;
    }

    @Override
    public void forwardSignalReceived(final Neuron from, final Double value) {
        signalReceived++;
        inputSignals.put(from, value);
        inputSignalsAverage += inputSignalsAverage;
        if (backwardConnections.size() == signalReceived) {
            double forwardInputToActivationFunction
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
    public void addForwardConnection(final Neuron neuron) {
        forwardConnections.add(neuron);
    }

    @Override
    public void addBackwardConnection(final Neuron neuron, final Double weight) {
        backwardConnections.put(neuron, weight);
        inputSignals.put(neuron, Double.NaN);
    }

    public static class Builder {

        private double bias = new Random().nextDouble();

        private ActivationFunction activationFunction;

        public Builder bias(final double bias) {
            this.bias = bias;
            return this;
        }

        public Builder activationFunction(final ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public ConnectedNeuron build() {
            if (activationFunction == null) {
                throw new RuntimeException("ActivationFunction need to be set in order to" +
                        " create a ConnectedNeuron");
            }
            return new ConnectedNeuron(
                    activationFunction,
                    bias);
        }
    }
}
