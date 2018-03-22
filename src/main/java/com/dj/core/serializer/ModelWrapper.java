package com.dj.core.serializer;


import com.dj.core.model.graph.Neuron;

import java.io.Serializable;
import java.util.List;


public class ModelWrapper implements Serializable {

    private final List<Neuron> inputLayer;
    private final List<Neuron> outputLayer;

    private ModelWrapper(final List<Neuron> inputLayer,
                        final List<Neuron> outputLayer) {
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
    }

    public List<Neuron> getInputLayer() {
        return inputLayer;
    }

    public List<Neuron> getOutputLayer() {
        return outputLayer;
    }

    public static class Builder {

        private List<Neuron> inputLayer;

        private List<Neuron> outputLayer;

        public Builder inputLayer(final List<Neuron> inputLayer) {
            this.inputLayer = inputLayer;
            return this;
        }

        public Builder outputLayer(final List<Neuron> outputLayer) {
            this.outputLayer = outputLayer;
            return this;
        }

        public ModelWrapper build() {
            return new ModelWrapper(this.inputLayer, this.outputLayer);
        }
    }
}
