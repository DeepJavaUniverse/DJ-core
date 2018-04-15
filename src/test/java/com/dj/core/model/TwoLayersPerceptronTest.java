package com.dj.core.model;

import com.dj.core.model.activation.Relu;
import com.dj.core.model.activation.Sigmoid;
import com.dj.core.model.graph.ConnectedNeuron;
import com.dj.core.model.graph.Context;
import com.dj.core.model.graph.InputNeuron;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertTrue;


public class TwoLayersPerceptronTest {

    private ConnectedNeuron outputNeuron;
    private ConnectedNeuron hiddenNeuron1;
    private ConnectedNeuron hiddenNeuron2;
    private InputNeuron inputFriend;
    private InputNeuron inputVodka;
    private InputNeuron inputSunny;

    @Before
    public void setUp() {
        Random random = new Random();
        double learningRate = 0.15;
        int batchSize = 4;
        inputFriend = new InputNeuron("friend");
        inputVodka = new InputNeuron("vodka");
        inputSunny = new InputNeuron("sunny");

        var context = new Context(learningRate, false, batchSize);

        hiddenNeuron1 = new ConnectedNeuron.Builder()
                .activationFunction(new Relu())
                .context(context)
                .name("hidden1")
                .build();
        inputFriend.connect(hiddenNeuron1, random.nextDouble());
        inputVodka.connect(hiddenNeuron1, random.nextDouble());
        inputSunny.connect(hiddenNeuron1, random.nextDouble());

        hiddenNeuron2 = new ConnectedNeuron.Builder()
                .activationFunction(new Relu())
                .context(context)
                .name("hidden2")
                .build();

        inputFriend.connect(hiddenNeuron2, random.nextDouble());
        inputVodka.connect(hiddenNeuron2, random.nextDouble());
        inputSunny.connect(hiddenNeuron2, random.nextDouble());

        outputNeuron = new ConnectedNeuron.Builder()
                .activationFunction(new Sigmoid())
                .context(context)
                .build();
        hiddenNeuron1.connect(outputNeuron, random.nextDouble());
        hiddenNeuron2.connect(outputNeuron, random.nextDouble());
    }

    @Test
    public void testTraining() throws Exception {
        var error = 0.;
        for (int i = 0; i < 600; i++) {
            error =
                    trainIteration(
                            new double[] { 1., 0., 1., 0. },
                            new double[] { 1., 1., 0., 1. },
                            new double[] { 1., 0., 0., 1. },
                            new double[] { 1., 0., 0., 1. })
                    + trainIteration(
                            new double[] { 0., 1., 1., 0. },
                            new double[] { 0., 1., 0., 0. },
                            new double[] { 1., 0., 1., 0. },
                            new double[] { 0., 0., 1., 0. });
            error = error / 8.;
            System.out.printf("ERROR: %s \n", error);
        }
        assertTrue(Math.abs(error) < 0.01);
    }

    private double trainIteration(double[] friendInput,
                                double[] vodkaInput,
                                double[] sunnyInput,
                                double[] expectedResult) {
        inputFriend.forwardSignalReceived(null, friendInput);
        inputVodka.forwardSignalReceived(null, vodkaInput);
        inputSunny.forwardSignalReceived(null, sunnyInput);
        var actualResult = outputNeuron.getForwardResult();

        var diff = 0.;
        final var errorDyTensor = new double[expectedResult.length];
        for (var i = 0; i < expectedResult.length; i++) {
            var localDiff = expectedResult[i] - actualResult[i];
            diff += Math.pow(localDiff, 2);
            errorDyTensor[i] = 2 * localDiff;
        }
        outputNeuron.backwardSignalReceived(errorDyTensor);

        IntStream.range(0, friendInput.length).forEach(i -> {
            System.out.printf("For: F: %f V: %f S: %f Expected: %f Actual: %f\n",
                    friendInput[i],
                    vodkaInput[i],
                    sunnyInput[i],
                    expectedResult[i],
                    actualResult[i]);
        });

        return diff;
    }
}
