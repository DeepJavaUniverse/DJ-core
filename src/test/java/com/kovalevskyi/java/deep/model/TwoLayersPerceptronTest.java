package com.kovalevskyi.java.deep.model;

import com.kovalevskyi.java.deep.model.activation.Relu;
import com.kovalevskyi.java.deep.model.activation.Sigmoid;
import com.kovalevskyi.java.deep.model.graph.ConnectedNeuron;
import com.kovalevskyi.java.deep.model.graph.InputNeuron;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static junit.framework.TestCase.assertTrue;


public class TwoLayersPerceptronTest {

    private ConnectedNeuron outputNeuron;
    private ConnectedNeuron hiddenNeuron1;
    private ConnectedNeuron hiddenNeuron2;
    private InputNeuron inputFriend;
    private InputNeuron inputVodka;
    private InputNeuron inputSunny;
    private ForkJoinPool forkJoinPool;

    @Before
    public void setUp() {
        forkJoinPool = new ForkJoinPool();
        Random random = new Random();
        double learningRate = 0.2;
        inputFriend = new InputNeuron("friend");
        inputVodka = new InputNeuron("vodka");
        inputSunny = new InputNeuron("sunny");

        hiddenNeuron1 = new ConnectedNeuron(
                Map.of(
                        inputFriend, random.nextDouble(),
                        inputVodka, random.nextDouble(),
                        inputSunny, random.nextDouble()
                ),
                random.nextDouble(),
                new Relu(),
                learningRate,
                forkJoinPool);
        inputFriend.addForwardConnection(hiddenNeuron1);
        inputVodka.addForwardConnection(hiddenNeuron1);
        inputSunny.addForwardConnection(hiddenNeuron1);

        hiddenNeuron2 = new ConnectedNeuron(
                Map.of(
                        inputFriend, random.nextDouble(),
                        inputVodka, random.nextDouble(),
                        inputSunny, random.nextDouble()
                ),
                random.nextDouble(),
                new Relu(),
                learningRate,
                forkJoinPool);
        inputFriend.addForwardConnection(hiddenNeuron2);
        inputVodka.addForwardConnection(hiddenNeuron2);
        inputSunny.addForwardConnection(hiddenNeuron2);

        outputNeuron = new ConnectedNeuron(
                Map.of(
                        hiddenNeuron1, random.nextDouble(),
                        hiddenNeuron2, random.nextDouble()
                ),
                random.nextDouble(),
                new Sigmoid(),
                learningRate,
                forkJoinPool);
        hiddenNeuron1.addForwardConnection(outputNeuron);
        hiddenNeuron2.addForwardConnection(outputNeuron);
    }

    @Test
    public void testTraining() throws Exception {
        double error = .0;
        for (int i = 0; i < 1000; i++) {
            error =
                    trainIteration(
                            1.,
                            1.,
                            1.,
                            1.)
                    + trainIteration(
                            1.,
                            1.,
                            0.,
                            0.)
                    + trainIteration(
                            1.,
                            0.,
                            1.,
                            1.)
                    + trainIteration(
                            0.,
                            1.,
                            1.,
                            1.)
                    + trainIteration(
                            0.,
                            0.,
                            1.,
                            0.)
                    + trainIteration(
                            0.,
                            1.,
                            0.,
                            0.)
                    + trainIteration(
                            1.,
                            0.,
                            0.,
                            0.)
                    + trainIteration(
                            0.,
                            0.,
                            0.,
                            0.);
            error = error / 8.;
            System.out.printf("ERROR: %s \n", error);
        }
        assertTrue(Math.abs(error) < 0.1);
    }

    private double trainIteration(double friendInput,
                                double vodkaInput,
                                double sunnyInput,
                                double expectedResult) throws Exception {
        inputFriend.forwardSignalReceived(null, friendInput);
        inputVodka.forwardSignalReceived(null, vodkaInput);
        inputSunny.forwardSignalReceived(null, sunnyInput);
        while (forkJoinPool.getActiveThreadCount() > 0) {
            Thread.sleep(5);
        }
        double actualResult = outputNeuron.getForwardResult();
        double errorDy = 2 * (expectedResult - actualResult);
        outputNeuron.backwardSignalReceived(errorDy);
        while (forkJoinPool.getActiveThreadCount() > 0) {
            Thread.sleep(5);
        }
        outputNeuron.forwardInvalidate();
        System.out.printf("For: F: %f V: %f S: %f Expected: %f Actual: %f\n",
                friendInput,
                vodkaInput,
                sunnyInput,
                expectedResult,
                actualResult);
        return expectedResult - actualResult;
    }
}