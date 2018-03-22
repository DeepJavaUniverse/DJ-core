package com.dj.core.model.graph;


public interface Neuron {

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

    default void connect(Neuron neuron, Double weight) {
        this.addForwardConnection(neuron);
        neuron.addBackwardConnection(this, weight);
    }

    void addForwardConnection(Neuron neuron);

    void addBackwardConnection(Neuron neuron, Double weight);

    double getForwardResult();
}
