package com.dj.core.optimizer;

import com.dj.core.model.graph.Context;
import com.dj.core.model.graph.Neuron;
import com.dj.core.model.loss.Loss;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;


public class SGDOptimizer implements Optimizer {

    private final Loss loss;
    private final int numberOfEpochsToTrain;
    private final OptimizerProgressListener lossCalculatedListener;
    private final double adjustLearningRateBy;
    private double currentLoss = Double.MAX_VALUE;

    public SGDOptimizer(final Loss loss,
                        final int numberOfEpochsToTrain,
                        final OptimizerProgressListener lossCalculatedListener,
                        final double adjustLearningRateBy) {
        this.loss = loss;
        this.numberOfEpochsToTrain = numberOfEpochsToTrain;
        this.lossCalculatedListener = lossCalculatedListener;
        this.adjustLearningRateBy = adjustLearningRateBy;
    }

    public SGDOptimizer(final Loss loss,
                        final int numberOfEpochsToTrain) {
        this(loss, numberOfEpochsToTrain, null, 1.0);
    }

    @Override
    public void train(
            final Context context,
            final List<Neuron> inputNeurons,
            final List<Neuron> outputNeurons,
            final double[][] inputData,
            final double[][] expectedResult,
            final double[][] inputTestData,
            final double[][] expectedTestResult) {
         IntStream.range(0, numberOfEpochsToTrain).forEach(epoch -> {
             if (lossCalculatedListener != null || adjustLearningRateBy != 1.) {
                 final double loss = calculateLoss(
                         context,
                         inputNeurons,
                         outputNeurons,
                         inputTestData,
                         expectedTestResult);
                 if (lossCalculatedListener != null) {
                     lossCalculatedListener.onProgress(
                             loss,
                             epoch,
                             numberOfEpochsToTrain);
                 }
                 if (adjustLearningRateBy != 1. && currentLoss < loss) {
                     System.out.printf("Loss have increased from: %f to %f\n", currentLoss, loss);
                     System.out.printf("Changing learning rate from: %f to %f\n",
                             context.getLearningRate(),
                             context.getLearningRate() * adjustLearningRateBy);
                     context.setLearningRate(context.getLearningRate() * adjustLearningRateBy);
                 }
                 currentLoss = loss;
             }
             final var batchSize = context.getBatchSize();
             IntStream.range(0, inputData.length / batchSize).forEach(batchIndex -> {
                 final var from = batchIndex * batchSize;
                 trainIteration(
                         inputNeurons,
                         outputNeurons,
                         fetchBatch(inputData, from, from + batchSize),
                         fetchBatch(expectedResult, from, from + batchSize));
             });
            }
         );
    }

    private void trainIteration(
            final List<Neuron> inputNeurons,
            final List<Neuron> outputNeurons,
            final double[][] inputDataBatch,
            final double[][] expectedResults) {
        IntStream.range(0, inputNeurons.size()).forEach(example -> {
            double[] inputSignals = IntStream.range(0, inputDataBatch.length)
                    .mapToDouble(i -> inputDataBatch[i][example])
                    .toArray();

            inputNeurons
                    .get(example)
                    .forwardSignalReceived(null, inputSignals);
        });
        IntStream.range(0, outputNeurons.size()).forEach(
                i -> {
                    final double[] actualValues = outputNeurons.get(i).getForwardResult();
                    outputNeurons
                            .get(i)
                            .backwardSignalReceived(
                                    loss
                                            .derivative(actualValues,
                                                        fetch1DSubArray(expectedResults, 0, expectedResults.length, i)));
                }
        );
    }

    private double calculateLoss(
            final Context context,
            final List<Neuron> inputNeurons,
            final List<Neuron> outputNeurons,
            final double[][] inputData,
            final double[][] expectedResults) {
        final var batchSize = context.getBatchSize();
        return IntStream
                .range(0, inputData.length / batchSize)
                .mapToDouble(batchIndex -> {
                    final var from = batchSize * batchIndex;
                    IntStream.range(0, inputData[batchIndex].length).forEach(i ->
                            inputNeurons
                                    .get(i)
                                    .forwardSignalReceived(
                                            null,
                                            fetch1DSubArray(inputData, from, from + batchSize, i)
                                    )
                    );
                    return IntStream
                            .range(0, expectedResults[batchIndex].length)
                            .mapToDouble(i ->
                                    loss.error(outputNeurons.get(i).getForwardResult()[0],
                                            expectedResults[batchIndex][i])
                            ).sum();
                }).average().getAsDouble();
    }

    private double[][] fetchBatch(final double[][] src, final int from, final int to) {
        return Arrays.copyOfRange(src, from, to);
    }

    private double[] fetch1DSubArray(final double[][] src, final int fromPos, final int toPos, final int columnIndex) {
        return IntStream.range(0, toPos)
                .filter(i -> i >= fromPos)
                .mapToDouble(i -> src[i][columnIndex])
                .toArray();
    }
}
