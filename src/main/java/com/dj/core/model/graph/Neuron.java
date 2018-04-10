package com.dj.core.model.graph;


import java.io.Serializable;

public interface Neuron extends Serializable {

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
    void forwardSignalReceived(Neuron from, double... values);

    void backwardSignalReceived(double... value);

    default void connect(Neuron neuron, Double weight) {
        this.addForwardConnection(neuron);
        neuron.addBackwardConnection(this, weight);
    }

    void addForwardConnection(Neuron neuron);

    void addBackwardConnection(Neuron neuron, Double weight);

    double[] getForwardResult();
}
