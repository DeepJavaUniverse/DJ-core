package com.kovalevskyi.java.deep.model;

import com.kovalevskyi.java.deep.model.activation.Sigmoid;
import com.kovalevskyi.java.deep.model.graph.ConnectedNeuron;
import com.kovalevskyi.java.deep.model.graph.InputNeuron;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class SingleLayerPerceptronTest {

    private ConnectedNeuron outputNeuron;
    private InputNeuron inputFriend;
    private InputNeuron inputVodka;
    private InputNeuron inputParty;

    @BeforeEach
    void setUp() {
        inputFriend = new InputNeuron("friend");
        inputVodka = new InputNeuron("vodka");
        inputParty = new InputNeuron("party");
    }

    @Test
    void testExecutionAfterInvalidatingTheState() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.,
                0.1);

        inputFriend.forwardSignalReceived(null, 1.);
        inputVodka.forwardSignalReceived(null, 1.);
        inputParty.forwardSignalReceived(null, 1.);
        Thread.sleep(500);
        assertTrue(outputNeuron.getForwardResult() > .7);
        outputNeuron.forwardInvalidate();

        inputFriend.forwardSignalReceived(null, 0.);
        inputVodka.forwardSignalReceived(null, 0.);
        inputParty.forwardSignalReceived(null, 0.);
        Thread.sleep(500);
        assertTrue(outputNeuron.getForwardResult() < .3);
    }

    @Test
    void testExecutionWhenResultShouldBeCloseTo1() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.,
                0.1);

        inputFriend.forwardSignalReceived(null, 1.);
        inputVodka.forwardSignalReceived(null, 1.);
        inputParty.forwardSignalReceived(null, 1.);
        Thread.sleep(500);
        assertTrue(outputNeuron.getForwardResult() > .7);
    }

    @Test
    void testExecutionWhenResultShouldBeCloseTo0() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.,
                0.1);

        inputFriend.forwardSignalReceived(null, 0.);
        inputVodka.forwardSignalReceived(null, 0.);
        inputParty.forwardSignalReceived(null, 0.);
        Thread.sleep(500);
        assertTrue(outputNeuron.getForwardResult() < .3);
    }

    @Test
    void testExecutionWhenResultShouldBeMoreThen0dot5() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.,
                0.1);

        inputFriend.forwardSignalReceived(null, 1.);
        inputVodka.forwardSignalReceived(null, 0.);
        inputParty.forwardSignalReceived(null, 1.);
        Thread.sleep(500);
        assertTrue(outputNeuron.getForwardResult() > .5);
    }

    @Test
    void testExecutionWhenResultShouldBeLessThen0dot5() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.,
                0.1);

        inputFriend.forwardSignalReceived(null,0.);
        inputVodka.forwardSignalReceived(null,0.);
        inputParty.forwardSignalReceived(null,1.);
        Thread.sleep(500);
        assertTrue(outputNeuron.getForwardResult() <= .5);
    }

    @Test
    void testTraining() throws Exception {
        Random random = new Random();
        initiateOutputNeuronWithWeights(
                random.nextDouble(),
                random.nextDouble(),
                random.nextDouble(),
                random.nextDouble(),
                0.1);

        for (int i = 0; i < 100; i++) {
            double error =
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
            System.out.println(outputNeuron.getBackwardConnections());
            System.out.println(outputNeuron.getBias());
        }
    }

    private double trainIteration(double friendInput,
                                double vodkaInput,
                                double partyInput,
                                double expectedResult) throws Exception {
        inputFriend.forwardSignalReceived(null, friendInput);
        inputVodka.forwardSignalReceived(null, vodkaInput);
        inputParty.forwardSignalReceived(null, partyInput);
        Thread.sleep(10);
        double actualResult = outputNeuron.getForwardResult();
        double dA = expectedResult - actualResult;
        dA = dA * dA * dA;
        outputNeuron.backwardSignalReceived(dA);
        Thread.sleep(10);
//        System.out.printf("TMP ERROR: %sd \n", dA / dA / dA);
        return expectedResult - actualResult;
    }

    private void initiateOutputNeuronWithWeights(double wFriend,
                                                 double wVodka,
                                                 double wParty,
                                                 double bias,
                                                 double learningRate) {
        outputNeuron = new ConnectedNeuron(
                Map.of(
                        inputFriend, wFriend,
                        inputVodka, wVodka,
                        inputParty, wParty
                ),
                -1.,
                new Sigmoid(),
                learningRate);
        inputFriend.addForwardConnection(outputNeuron);
        inputVodka.addForwardConnection(outputNeuron);
        inputParty.addForwardConnection(outputNeuron);
    }
}