package com.kovalevskyi.java.deep.model;

import com.kovalevskyi.java.deep.model.activation.Sigmoid;
import com.kovalevskyi.java.deep.model.graph.ConnectedNeuron;
import com.kovalevskyi.java.deep.model.graph.InputNeuron;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SingleLayerPerceptronTest {

    private ConnectedNeuron outputNeuron;
    private InputNeuron inputFriend;
    private InputNeuron inputVodka;
    private InputNeuron inputParty;

    @BeforeEach
    void setUp() {
        inputFriend = new InputNeuron();
        inputVodka = new InputNeuron();
        inputParty = new InputNeuron();
        outputNeuron = new ConnectedNeuron(
                Map.of(
                        inputFriend, 1.,
                        inputVodka, 1.,
                        inputParty, 1.
                ),
                -1.,
                new Sigmoid());
        inputFriend.addForwardConnection(outputNeuron);
        inputVodka.addForwardConnection(outputNeuron);
        inputParty.addForwardConnection(outputNeuron);
    }

    @Test
    void testExecutionAfterInvalidatingTheState() throws Exception {
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
        inputFriend.forwardSignalReceived(null, 1.);
        inputVodka.forwardSignalReceived(null, 1.);
        inputParty.forwardSignalReceived(null, 1.);
        Thread.sleep(500);
        assertTrue(outputNeuron.getForwardResult() > .7);
    }

    @Test
    void testExecutionWhenResultShouldBeCloseTo0() throws Exception {
        inputFriend.forwardSignalReceived(null, 0.);
        inputVodka.forwardSignalReceived(null, 0.);
        inputParty.forwardSignalReceived(null, 0.);
        Thread.sleep(500);
        assertTrue(outputNeuron.getForwardResult() < .3);
    }

    @Test
    void testExecutionWhenResultShouldBeMoreThen0dot5() throws Exception {
        inputFriend.forwardSignalReceived(null, 1.);
        inputVodka.forwardSignalReceived(null, 0.);
        inputParty.forwardSignalReceived(null, 1.);
        Thread.sleep(500);
        assertTrue(outputNeuron.getForwardResult() > .5);
    }

    @Test
    void testExecutionWhenResultShouldBeLessThen0dot5() throws Exception {
        inputFriend.forwardSignalReceived(null,0.);
        inputVodka.forwardSignalReceived(null,0.);
        inputParty.forwardSignalReceived(null,1.);
        Thread.sleep(500);
        assertTrue(outputNeuron.getForwardResult() <= .5);
    }
}