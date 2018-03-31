package com.dj.core.model;

import com.dj.core.model.graph.Context;
import com.dj.core.model.graph.InputNeuron;
import com.dj.core.model.activation.Sigmoid;
import com.dj.core.model.graph.ConnectedNeuron;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

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
    public void test2Execution() throws Exception {
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
        var random = new Random();
        initiateOutputNeuronWithWeights(
                random.nextDouble(),
                random.nextDouble(),
                random.nextDouble(),
                random.nextDouble(),
                0.05);
        double error = 0.;
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
        var actualResult = outputNeuron.getForwardResult();
        var dA = expectedResult - actualResult;
        dA = dA * dA * dA;
        outputNeuron.backwardSignalReceived(dA);
        return expectedResult - actualResult;
    }

    private void initiateOutputNeuronWithWeights(double wFriend,
                                                 double wVodka,
                                                 double wSunny,
                                                 double bias,
                                                 double learningRate) {
        var context = new Context(learningRate, false);
        outputNeuron
                = new ConnectedNeuron.Builder()
                    .bias(bias)
                    .activationFunction(new Sigmoid())
                    .context(context)
                    .build();
        inputFriend.connect(outputNeuron, wFriend);
        inputVodka.connect(outputNeuron, wVodka);
        inputSunny.connect(outputNeuron, wSunny);
    }
}