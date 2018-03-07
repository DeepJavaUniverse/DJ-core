package com.kovalevskyi.java.deep.core.model.optimizer;

import com.kovalevskyi.java.deep.core.model.graph.Neuron;
import com.kovalevskyi.java.deep.core.model.loss.Loss;

import java.util.List;

public interface Optimizer {

    void train(final List<Neuron> inputNeurons,
               final List<Neuron> outputNeurons,
               final Loss lossFunction,
               double[][] inputData,
               double[][] expectedResult);
}
