package com.kovalevskyi.java.deep.core.model;

import com.kovalevskyi.java.deep.core.model.activation.Sigmoid;
import com.kovalevskyi.java.deep.core.model.graph.ConnectedNeuron;
import com.kovalevskyi.java.deep.core.model.graph.InputNeuron;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static junit.framework.TestCase.assertTrue;


public class SingleLayerPerceptronTest {

    private ConnectedNeuron outputNeuron;
    private InputNeuron inputFriend;
    private InputNeuron inputVodka;
    private InputNeuron inputSunny;

    @Before
    public void setUp() {
        inputFriend = new InputNeuron("friend");
        inputVodka = new InputNeuron("vodka");
        inputSunny = new InputNeuron("sunny");
    }

    @Test
    public void testExecutionAfterInvalidatingTheState() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.,
                0.1);

        inputFriend.forwardSignalReceived(null, 1.);
        inputVodka.forwardSignalReceived(null, 1.);
        inputSunny.forwardSignalReceived(null, 1.);
        assertTrue(outputNeuron.getForwardResult() > .7);
        outputNeuron.forwardInvalidate();

        inputFriend.forwardSignalReceived(null, 0.);
        inputVodka.forwardSignalReceived(null, 0.);
        inputSunny.forwardSignalReceived(null, 0.);
        assertTrue(outputNeuron.getForwardResult() < .3);
    }

    @Test
    public void testExecutionWhenResultShouldBeCloseTo1() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.,
                0.1);

        inputFriend.forwardSignalReceived(null, 1.);
        inputVodka.forwardSignalReceived(null, 1.);
        inputSunny.forwardSignalReceived(null, 1.);
        assertTrue(outputNeuron.getForwardResult() > .7);
    }

    @Test
    public void testExecutionWhenResultShouldBeCloseTo0() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.,
                0.1);

        inputFriend.forwardSignalReceived(null, 0.);
        inputVodka.forwardSignalReceived(null, 0.);
        inputSunny.forwardSignalReceived(null, 0.);
        assertTrue(outputNeuron.getForwardResult() < .3);
    }

    @Test
    public void testExecutionWhenResultShouldBeMoreThen0dot5() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.,
                0.1);

        inputFriend.forwardSignalReceived(null, 1.);
        inputVodka.forwardSignalReceived(null, 0.);
        inputSunny.forwardSignalReceived(null, 1.);
        assertTrue(outputNeuron.getForwardResult() > .5);
    }

    @Test
    public void testExecutionWhenResultShouldBeLessThen0dot5() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.,
                0.1);

        inputFriend.forwardSignalReceived(null,0.);
        inputVodka.forwardSignalReceived(null,0.);
        inputSunny.forwardSignalReceived(null,1.);
        assertTrue(outputNeuron.getForwardResult() <= .5);
    }

    @Test
    public void testTraining() throws Exception {
        Random random = new Random();
        initiateOutputNeuronWithWeights(
                random.nextDouble(),
                random.nextDouble(),
                random.nextDouble(),
                random.nextDouble(),
                0.05);
        double error = 0;
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
                            1.)
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
        assertTrue(error < 0.1);
    }

    private double trainIteration(double friendInput,
                                double vodkaInput,
                                double sunnyInput,
                                double expectedResult) throws Exception {
        inputFriend.forwardSignalReceived(null, friendInput);
        inputVodka.forwardSignalReceived(null, vodkaInput);
        inputSunny.forwardSignalReceived(null, sunnyInput);
        double actualResult = outputNeuron.getForwardResult();
        double dA = expectedResult - actualResult;
        dA = dA * dA * dA;
        outputNeuron.backwardSignalReceived(dA);
        outputNeuron.forwardInvalidate();
        return expectedResult - actualResult;
    }

    private void initiateOutputNeuronWithWeights(double wFriend,
                                                 double wVodka,
                                                 double wSunny,
                                                 double bias,
                                                 double learningRate) {
        outputNeuron
                = new ConnectedNeuron.Builder()
                    .bias(bias)
                    .activationFunction(new Sigmoid())
                    .learningRate(learningRate)
                    .build();
        inputFriend.connect(outputNeuron, wFriend);
        inputVodka.connect(outputNeuron, wVodka);
        inputSunny.connect(outputNeuron, wSunny);
    }
}