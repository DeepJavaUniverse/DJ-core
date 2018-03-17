package com.kovalevskyi.java.deep.core.optimizer;

import com.kovalevskyi.java.deep.core.model.graph.Neuron;
import com.kovalevskyi.java.deep.core.model.loss.Loss;

import java.util.List;


public interface Optimizer {

    default void train(final List<Neuron> inputNeurons,
               final List<Neuron> outputNeurons,
               double[][] inputData,
               double[][] expectedResult) {
        train(inputNeurons,
                outputNeurons,
                inputData,
                expectedResult,
                inputData,
                expectedResult);
    }

    void train(final List<Neuron> inputNeurons,
               final List<Neuron> outputNeurons,
               double[][] inputData,
               double[][] expectedResult,
               double[][] inputTestData,
               double[][] expectedTestResult);
}
