package com.dj.core.model.graph;

import com.dj.core.model.activation.ActivationFunction;
import com.google.common.util.concurrent.AtomicDouble;

import java.util.*;

public class ConnectedNeuron implements Neuron {

    private final Context context;

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

    private final String name;

    /**
     * inputSignals is used to store the signals from other Neurons. Keys in this Map should be absolutely identical to
     * the keys in the {@link #backwardConnections}. As soon as all the signals received Neuron can start processing
     * them.
     */
    private final Map<Neuron, Double> inputSignals = new HashMap<>();

    private final AtomicDouble bias;

    /**
     * Amount of the signals that has been received already. As soon as this number reaches the size of the
     * {@link #inputSignals} map the neuron is ready to start processing input signals and send signal forward.
     */
    private volatile int signalReceived;

    /**
     * Stores result of the latest signal that was send from the Neuron to other Neurons. This is mostly needed for the
     * output Neurons, since they do not have any other Neurons to send signal to and in the same time there should be
     * the way of getting this value.
     */
    private volatile double forwardResult;
    
    private volatile double inputSignalsSum;

    private volatile double inputSignalsAverage;

    private volatile double forwardInputToActivationFunction;

    private ConnectedNeuron(
            final ActivationFunction activationFunction,
            final String name,
            final double bias,
            final Context context) {
        this.activationFunction = activationFunction;
        this.context = context;
        this.name = name;
        this.bias = new AtomicDouble(bias);
    }

    @Override
    public double getForwardResult() {
        return forwardResult;
    }

    @Override
    public void forwardSignalReceived(final Neuron from, final Double value) {
        signalReceived++;
        inputSignals.put(from, value);
        inputSignalsSum += value;
        // The following if is the check weather current signal was the last remaining signal to receive. And if so and
        // all incoming signals have been received the Neuron can start processing them and issue new signal himself.
        if (backwardConnections.size() == signalReceived) {
            // 4 steps need to happen when Neuron processes the input signals:
            // 1. Calculate input = W * X + b
            // 2. Calculate output = f(input), where f is activation function
            // 3. Send output to other neurons
            // 4. invalidate state

            // Step #1
            // Calculating W * X + b - sum of all input signals, each signal multiplied on the corresponding weight.
            // Bias is added at the end.
            forwardInputToActivationFunction
                    = backwardConnections
                        .entrySet()
                        .stream()
                        .mapToDouble(connection ->
                                // inputSignals store the actual signal, while connection.getValue() gives you the
                                // weight that the signal should be multiplied to. Therefore this part is X * W.
                                inputSignals.get(connection.getKey())
                                        * connection.getValue())
                        .sum() + bias.get();
            if (context.isDebugMode()) {
                if (brokenValue(forwardInputToActivationFunction)) {
                    throw new RuntimeException("Forward input to activation function is broken");
                }
            }

            // Step #2
            double signalToSend
                    = activationFunction.forward(
                            forwardInputToActivationFunction);
            forwardResult = signalToSend;

            // Step #3 Since signal is calculated now we can send it to other neurons.
            forwardConnections
                    .stream()
                    .forEach(connection ->
                        connection
                                .forwardSignalReceived(
                                        ConnectedNeuron.this,
                                        signalToSend)
                    );

            // Step #4
            inputSignalsAverage
                    = inputSignalsSum / (double) signalReceived;
            inputSignalsSum = 0.;
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
        if (context.isDebugMode()) {
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
                    final double newWeight = weight + inputSignals.get(conn) * dz * context.getLearningRate();
                    if (context.isDebugMode()) {
                        if (brokenValue(newWeight)) {
                            throw new RuntimeException("Updated connection weight is broken");
                        }
                    }
                    return newWeight;
                }
            ));
        bias.addAndGet(inputSignalsAverage * dz * context.getLearningRate());
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

        private Context context;

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

        public Builder context(final Context context) {
            this.context = context;
            return this;
        }
        
        public ConnectedNeuron build() {
            if (activationFunction == null) {
                throw new RuntimeException("ActivationFunction need to be set in order to" +
                        " create a ConnectedNeuron");
            }
            if (context == null) {
                throw new RuntimeException("Context need to be set in order to" +
                        " create a ConnectedNeuron");
            }
            return new ConnectedNeuron(
                    activationFunction,
                    name,
                    bias,
                    context);
        }
    }
}
