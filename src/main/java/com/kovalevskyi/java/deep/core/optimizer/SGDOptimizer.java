package com.kovalevskyi.java.deep.core.optimizer;

import com.kovalevskyi.java.deep.core.model.graph.Neuron;
import com.kovalevskyi.java.deep.core.model.loss.Loss;

import java.util.List;
import java.util.stream.IntStream;

public class SGDOptimizer implements Optimizer {

    private final Loss loss;

    public SGDOptimizer(final Loss loss) {
        this.loss = loss;
    }

    @Override
    public void train(
            final List<Neuron> inputNeurons,
            final List<Neuron> outputNeurons,
            final double[][] inputData,
            final double[][] expectedResult) {
        // TODO
    }

    private static void trainIteration(
            final List<Neuron> inputNeurons,
            final List<Neuron> outputNeurons,
            final Loss lossFunction,
            final double[] inputData,
            final double[] expectedResults) {
        IntStream.range(0, inputData.length).forEach(i ->
            inputNeurons
                    .get(i)
                    .forwardSignalReceived(null, inputData[i])
        );
        IntStream.range(0, outputNeurons.size()).forEach(
                i -> {
                    final double actualValue = outputNeurons.get(i).getForwardResult();
                    final double expectedResult = expectedResults[i];
                    outputNeurons
                            .get(i)
                            .backwardSignalReceived(
                                    lossFunction
                                            .derivative(actualValue, expectedResult));
                }
        );
    }
}
