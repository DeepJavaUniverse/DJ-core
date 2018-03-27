package com.dj.core.serializer;


import com.dj.core.model.graph.Context;
import com.dj.core.model.graph.Neuron;

import java.io.Serializable;
import java.util.List;


public class ModelWrapper implements Serializable {

    private final Context context;
    private final List<Neuron> inputLayer;
    private final List<Neuron> outputLayer;

    private ModelWrapper(final Context context,
                         final List<Neuron> inputLayer,
                         final List<Neuron> outputLayer) {
        this.context = context;
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
    }

    public Context getContext() {
        return context;
    }

    public List<Neuron> getInputLayer() {
        return inputLayer;
    }

    public List<Neuron> getOutputLayer() {
        return outputLayer;
    }

    public static class Builder {

        private Context context;

        private List<Neuron> inputLayer;

        private List<Neuron> outputLayer;

        public Builder context(final Context context) {
            this.context = context;
            return this;
        }

        public Builder inputLayer(final List<Neuron> inputLayer) {
            this.inputLayer = inputLayer;
            return this;
        }

        public Builder outputLayer(final List<Neuron> outputLayer) {
            this.outputLayer = outputLayer;
            return this;
        }

        public ModelWrapper build() {
            return new ModelWrapper(this.context, this.inputLayer, this.outputLayer);
        }
    }
}
