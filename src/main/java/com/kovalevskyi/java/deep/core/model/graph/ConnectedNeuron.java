package com.kovalevskyi.java.deep.core.model.graph;

import com.kovalevskyi.java.deep.core.model.activation.ActivationFunction;

import java.util.*;

public class ConnectedNeuron implements Neuron {

    private final ActivationFunction activationFunction;

    /**
     * Represents the connections from the neuron to the neurons that it receives signals from. For example in the
     * following network:
     * NeuronA ___
     *            \ weight1 = -0.1
     *             \
     * weight2 = 0.1\
     * NeuronB ------ NeuronD
     *              /
     *             /
     *            /  weight3 = 0.8
     * NeuronC ---
     * backwardConnections map will looks like this:
     * NeuronA => -0.1
     * NeuronB => 0.1
     * NeuronC => 0.8
     */
    private final Map<Neuron, Double> backwardConnections = new HashMap<>();

    /**
     * Represents set of the Neurons to which current neuron sends signals to. There is no need in weights here.
     */
    private final Set<Neuron> forwardConnections = new HashSet<>();

    /**
     * inputSignals is used to store the signals from other Neurons. Keys in this Map should be absolutely identical to
     * the keys in the {@link #backwardConnections}. As soon as all the signals received Neuron can start processing
     * them.
     */
    private final Map<Neuron, Double> inputSignals = new HashMap<>();

    /**
     * Amount of the signals that has been received already. As soon as this number reaches the size of the
     * {@link #inputSignals} map the neuron is ready to start processing input signals and send signal forward.
     */
    private volatile int signalReceived;
    private final double bias;

    /**
     * Stores result of the latest signal that was send from the Neuron to other Neurons. This is mostly needed for the
     * output Neurons, since they do not have any other Neurons to send signal to and in the same time there should be
     * the way of getting this value.
     */
    private volatile double forwardResult;

    private ConnectedNeuron(
            final ActivationFunction activationFunction,
            final double bias) {
        this.activationFunction = activationFunction;
        this.bias = bias;
    }

    public double getForwardResult() {
        return forwardResult;
    }

    @Override
    public void forwardSignalReceived(final Neuron from, final Double value) {
        signalReceived++;
        inputSignals.put(from, value);
        if (backwardConnections.size() == signalReceived) {
            double forwardInputToActivationFunction
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

            forwardConnections
                    .stream()
                    .forEach(connection -> {
                        connection
                                .forwardSignalReceived(
                                        ConnectedNeuron.this,
                                        signalToSend);
                    });
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
