package com.kovalevskyi.java.deep.core.optimizer;

import com.kovalevskyi.java.deep.core.model.graph.Neuron;
import com.kovalevskyi.java.deep.core.model.loss.Loss;

import java.util.List;
import java.util.stream.IntStream;

public class SGDOptimizer implements Optimizer {

    private final Loss loss;
    private final int numberOfEpochsToTrain;
    private final LossCalculatedListener lossCalculatedListener;

    public SGDOptimizer(final Loss loss,
                        final int numberOfEpochsToTrain,
                        final LossCalculatedListener lossCalculatedListener) {
        this.loss = loss;
        this.numberOfEpochsToTrain = numberOfEpochsToTrain;
        this.lossCalculatedListener = lossCalculatedListener;
    }

    public SGDOptimizer(final Loss loss,
                        final int numberOfEpochsToTrain) {
        this(loss, numberOfEpochsToTrain, null);
    }

    @Override
    public void train(
            final List<Neuron> inputNeurons,
            final List<Neuron> outputNeurons,
            final double[][] inputData,
            final double[][] expectedResult,
            final double[][] inputTestData,
            final double[][] expectedTestResult) {
         IntStream.range(0, numberOfEpochsToTrain).forEach(i -> {
                trainIteration(
                        inputNeurons,
                        outputNeurons,
                        inputData[i],
                        expectedResult[i]);
                if (lossCalculatedListener != null) {
                    final double loss = calculateLoss(
                            inputNeurons,
                            outputNeurons,
                            inputTestData,
                            expectedTestResult);
                    lossCalculatedListener.onLossCalculated(loss);
                }
            }
         );
    }

    private void trainIteration(
            final List<Neuron> inputNeurons,
            final List<Neuron> outputNeurons,
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
                                    loss
                                            .derivative(actualValue, expectedResult));
                }
        );
    }

    private double calculateLoss(
            final List<Neuron> inputNeurons,
            final List<Neuron> outputNeurons,
            final double[][] inputData,
            final double[][] expectedResults) {
        IntStream.range(0, inputData.length).forEach({
            IntStream.range(0, inputData.length).forEach(i ->
                    inputNeurons
                            .get(i)
                            .forwardSignalReceived(null, inputData[i])
            );
            return IntStream.range(0, expectedResults.length).mapToDouble(i ->
                    loss.error(
                            outputNeurons.get(i).getForwardResult(),
                            expectedResults[i])
            ).sum();
        });
    }
}
