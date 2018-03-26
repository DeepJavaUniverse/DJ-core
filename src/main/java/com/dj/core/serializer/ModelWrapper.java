package com.dj.core.serializer;


import com.dj.core.model.graph.Context;
import com.dj.core.model.graph.Neuron;

import java.io.Serializable;
import java.util.List;


public class ModelWrapper implements Serializable {

    private final List<Neuron> inputLayer;
    private final List<Neuron> outputLayer;
    private final Context context;

    private ModelWrapper(final List<Neuron> inputLayer,
                         final List<Neuron> outputLayer,
                         final Context context) {
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.context = context;
    }

    public List<Neuron> getInputLayer() {
        return inputLayer;
    }

    public List<Neuron> getOutputLayer() {
        return outputLayer;
    }

    public Context getContext() {
        return context;
    }

    public static class Builder {

        private List<Neuron> inputLayer;

        private List<Neuron> outputLayer;

        private Context context;

        public Builder inputLayer(final List<Neuron> inputLayer) {
            this.inputLayer = inputLayer;
            return this;
        }

        public Builder outputLayer(final List<Neuron> outputLayer) {
            this.outputLayer = outputLayer;
            return this;
        }

        public Builder context(final Context context) {
            this.context = context;
            return this;
        }

        public ModelWrapper build() {
            return new ModelWrapper(this.inputLayer, this.outputLayer, context);
        }
    }
}
