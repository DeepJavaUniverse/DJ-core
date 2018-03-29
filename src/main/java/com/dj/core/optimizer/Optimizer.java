package com.dj.core.optimizer;

import com.dj.core.model.graph.Context;
import com.dj.core.model.graph.Neuron;

import java.util.List;


public interface Optimizer {

    default void train(
            final Context context,
            final List<Neuron> inputNeurons,
            final List<Neuron> outputNeurons,
            final double[][] inputData,
            final double[][] expectedResult) {
        train(context,
                inputNeurons,
                outputNeurons,
                inputData,
                expectedResult,
                inputData,
                expectedResult);
    }

    void train(
            final Context context,
            final List<Neuron> inputNeurons,
            final List<Neuron> outputNeurons,
            double[][] inputData,
            double[][] expectedResult,
            double[][] inputTestData,
            double[][] expectedTestResult);
}
