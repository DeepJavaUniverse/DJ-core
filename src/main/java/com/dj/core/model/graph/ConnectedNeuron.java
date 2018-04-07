package com.dj.core.model.graph;

import com.dj.core.model.activation.ActivationFunction;
import com.dj.core.model.visitor.SimpleRealMatrixChangingVisitor;
import com.dj.core.utils.ArrayUtils;
import com.google.common.util.concurrent.AtomicDouble;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;
import org.apache.commons.math3.linear.*;

import java.util.*;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class ConnectedNeuron implements Neuron {

    private final static int VECTOR_ROWS_COUNT = 1;

    private final static int VECTOR_ROW_INDEX = 0;

    private final Context context;

    private final ActivationFunction activationFunction;

    /**
     * Represents a relation between a neuron and a neuron index (it's position). It's necessary to store pairs
     * (Neuron, Index) to have a relation between neurons and their weights (in the {@link #backwardConnections}) and
     * their input signals (in the {@link #inputSignals}).
     */
    private final Map<Neuron, Integer> neuronIndexes = new HashMap<>();

    /**
     * Represents the connections (their weights) from the neuron to the neurons that it receives signals from.
     * For example in the following network:
     * NeuronA ___
     *            \ weight1 = -0.1
     *             \
     * weight2 = 0.1\
     * NeuronB ------ NeuronD
     *              /
     *             /
     *            /  weight3 = 0.8
     * NeuronC ---
     *
     * backwardConnections tensor will looks like this: [ -0.1, 0.1, 0.8 ]
     */
    private RealMatrix backwardConnections = new Array2DRowRealMatrix();

    /**
     * Represents set of the Neurons to which current neuron sends signals to. There is no need in weights here.
     */
    private final Set<Neuron> forwardConnections = new HashSet<>();

    private final String name;

    /**
     * inputSignals is used to store the signals from other Neurons. Values in this tensor should be stored according to
     * {@link #neuronIndexes}. If the neuron NeuronX was added to the {@link #neuronIndexes} with the index 0, the
     * {@link #inputSignals} will store the NeuronX's input signal at the position (rowIndex, 0).
     * As soon as all the signals received Neuron can start processing them.
     */
    private RealMatrix inputSignals = new Array2DRowRealMatrix();

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
    private volatile RealMatrix forwardResult;

    private volatile double inputSignalsSum;

    private volatile double inputSignalsAverage;

    private volatile RealMatrix forwardInputToActivationFunction;

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
    public double[] getForwardResult() {
        return forwardResult.getRow(VECTOR_ROW_INDEX);
    }

    @Override
    public void forwardSignalReceived(final Neuron from, final Double... values) {
        signalReceived++;
        var valuesPrimitives = ArrayUtils.toPrimitive(values);
        inputSignals.setColumn(neuronIndex(from), valuesPrimitives);
        inputSignalsSum += DoubleStream.of(valuesPrimitives)
                .average()
                .orElse(0.);

        // The following if is the check weather current signal was the last remaining signal to receive. And if so and
        // all incoming signals have been received the Neuron can start processing them and issue new signal himself.
        if (backwardConnections.getColumnDimension() == signalReceived) {
            // 4 steps need to happen when Neuron processes the input signals:
            // 1. Calculate input = W * X + b
            // 2. Calculate output = f(input), where f is activation function
            // 3. Send output to other neurons
            // 4. invalidate state

            // Step #1
            // Calculating W * X + b - sum of all input signals, each signal multiplied on the corresponding weight.
            // Bias is added at the end.
            forwardInputToActivationFunction = backwardConnections
                    .multiply(inputSignals.transpose())
                    .scalarAdd(bias.get());

            if (context.isDebugMode()) {
                DoubleStream.of(forwardInputToActivationFunction.getRow(VECTOR_ROW_INDEX))
                        .forEach(value -> {
                            if (brokenValue(value)) {
                                throw new RuntimeException("Forward input to activation function is broken");
                            }
                        });
            }

            // Step #2
            final var signalToSend = forwardInputToActivationFunction;
            signalToSend.walkInRowOrder(new SimpleRealMatrixChangingVisitor() {
                @Override
                public double visit(int row, int column, double value) {
                    return activationFunction.forward(value);
                }
            });

            forwardResult = signalToSend;

            // Step #3 Since signal is calculated now we can send it to other neurons.
            forwardConnections
                    .stream()
                    .forEach(connection ->
                            connection
                                    .forwardSignalReceived(
                                            ConnectedNeuron.this,
                                            ArrayUtils.toObject(signalToSend.getRow(VECTOR_ROW_INDEX)))
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
        var forwardInputToActivationFunctionAverage
                = DoubleStream.of(forwardInputToActivationFunction.getRow(VECTOR_ROW_INDEX))
                    .average()
                    .orElse(0.);
        final var derivative
                = activationFunction.backward(forwardInputToActivationFunctionAverage);

        final var dz = derivative * error;
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

        final var dzLearningRate = dz * context.getLearningRate();
        inputSignals = inputSignals.scalarMultiply(dzLearningRate);
        final var inputSignalsAverageVector = new Array2DRowRealMatrix(VECTOR_ROWS_COUNT, inputSignals.getColumnDimension());
        IntStream.range(0, inputSignals.getColumnDimension()).forEach(i -> {
            final var entry = DoubleStream.of(inputSignals.getColumn(i)).average().orElse(0.);
            inputSignalsAverageVector.setEntry(VECTOR_ROW_INDEX, i, entry);
        });

        backwardConnections = backwardConnections.add(inputSignalsAverageVector);

        bias.addAndGet(inputSignalsAverage * dz * context.getLearningRate());
        neuronIndexes
                .entrySet()
                .stream()
                .forEach(neuronIndex ->
                        neuronIndex.getKey().backwardSignalReceived(
                                backwardConnections
                                        .getEntry(VECTOR_ROW_INDEX, neuronIndex.getValue()) * dz));
    }

    @Override
    public void addForwardConnection(final Neuron neuron) {
        forwardConnections.add(neuron);
    }

    @Override
    public void addBackwardConnection(final Neuron neuron, final Double weight) {
        int nextIndex = neuronIndexes.size();
        neuronIndexes.put(neuron, nextIndex);

        var oldTensor = backwardConnections;
        backwardConnections = new Array2DRowRealMatrix(VECTOR_ROWS_COUNT, nextIndex + 1);
        if (oldTensor.getColumnDimension() != 0)
            backwardConnections.setSubMatrix(oldTensor.getData(), 0, 0);
        backwardConnections.setEntry(VECTOR_ROW_INDEX, nextIndex, weight);

        oldTensor= inputSignals;
        inputSignals = new Array2DRowRealMatrix(context.getBatchSize(), nextIndex + 1);
        if (oldTensor.getColumnDimension() != 0)
            inputSignals.setSubMatrix(oldTensor.getData(), 0, 0);
        IntStream.range(0, context.getBatchSize()).forEach(i -> {
            inputSignals.setEntry(i, nextIndex, Double.NaN);
        });
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

    private int neuronIndex(final Neuron n) {
        return neuronIndexes.get(n);
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
