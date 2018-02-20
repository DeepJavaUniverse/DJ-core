package com.kovalevskyi.java.deep.model.graph;

import java.util.Map;
import java.util.Set;

public abstract class Neuron {

    public abstract void forwardInvalidate();

    /**
     * Should be called when a Neuron receives input signal from the connected neuron.
     * For example let's look on the following network:
     * NeuronA
     *         \
     *          \
     *           \
     * NeuronB --- NeuronD
     *           /
     *          /
     *         /
     * NeuronC
     *
     * If NeuronA or NeuronB or NeuronC sends signal to NeuronD, the method should be
     * called.
     *
     * @param from , Neuron that sends the signal.
     */
    public abstract void forwardSignalReceived(Neuron from, Double value);

    public abstract void backwardSignalReceived(Double value);

    public abstract void connect(Neuron neuron, Double weight);

    abstract void addBackwardConnection(Neuron neuron, Double weight);
}
