package com.dj.core.optimizer;

import com.dj.core.model.graph.Context;
import com.dj.core.model.graph.Neuron;
import com.dj.core.model.loss.Loss;

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
             IntStream.range(0, inputData.length).forEach(index -> {
                 trainIteration(
                         inputNeurons,
                         outputNeurons,
                         inputData[index],
                         expectedResult[index]);
             });
            }
         );
    }

    private void trainIteration(
            final List<Neuron> inputNeurons,
            final List<Neuron> outputNeurons,
            final double[] inputData,
            final double[] expectedResults) {
        IntStream.range(0, inputData.length).forEach(example ->
            inputNeurons
                    .get(example)
                    .forwardSignalReceived(null, inputData[example])
        );
        IntStream.range(0, outputNeurons.size()).forEach(
                i -> {
//                    final double actualValue = outputNeurons.get(i).getForwardResult();
//                    final double expectedResult = expectedResults[i];
//                    outputNeurons
//                            .get(i)
//                            .backwardSignalReceived(
//                                    loss
//                                            .derivative(actualValue, expectedResult));
                }
        );
    }

    private double calculateLoss(
            final List<Neuron> inputNeurons,
            final List<Neuron> outputNeurons,
            final double[][] inputData,
            final double[][] expectedResults) {
        return 0.;
//        return IntStream
//                .range(0, inputData.length)
//                .mapToDouble(exampleIndex -> {
//                    IntStream.range(0, inputData[exampleIndex].length).forEach(i ->
//                            inputNeurons
//                                    .get(i)
//                                    .forwardSignalReceived(
//                                            null,
//                                            inputData[exampleIndex][i])
//                    );
//                    return IntStream
//                            .range(0, expectedResults[exampleIndex].length)
//                            .mapToDouble(i ->
//                                loss.error(
//                                        outputNeurons
//                                                .get(i)
//                                                .getForwardResult(),
//                                        expectedResults[exampleIndex][i])
//                    ).sum();
//        }).average().getAsDouble();
    }
}
