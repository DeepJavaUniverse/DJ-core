package com.kovalevskyi.java.deep.model.graph;

import java.util.Map;
import java.util.Set;

public interface Neuron {

    void forwardInvalidate();

    Set<Neuron> backwardConnections();

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
    void forwardSignalReceived(Neuron from, Double value);

    void backwardSignalReceived(Double value);

    void addForwardConnection(Neuron neuron);

    void addBackwardConnection(Neuron neuron, Double weight);

    Map<Neuron, Double> getBackwardConnections();

    double getBias();

}
