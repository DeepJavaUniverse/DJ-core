package com.kovalevskyi.java.deep.model;

import com.kovalevskyi.java.deep.model.activation.Sigmoid;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SingleLayerPerceptronTest {

    private Neuron outputNeuron;
    private InputNeuron inputFriend = new InputNeuron();
    private InputNeuron inputVodka = new InputNeuron();
    private InputNeuron inputPartie = new InputNeuron();

    @BeforeEach
    void setUp() {
        outputNeuron = new ConnectedNeuron(
                Map.of(
                        inputFriend, 1.,
                        inputVodka, 1.,
                        inputPartie, 1.
                ),
                -1.,
                new Sigmoid(),
                true);
    }

    @Test
    void testExecutionAfterInvalidatingTheState() throws Exception {
         inputFriend.setValue(1.);
         inputVodka.setValue(1.);
         inputPartie.setValue(1.);
         assertTrue(outputNeuron.call() > .7);
         outputNeuron.invalidate();

         inputFriend.setValue(0.);
         inputVodka.setValue(0.);
         inputPartie.setValue(0.);
         assertTrue(outputNeuron.call() < .3);
    }

    @Test
    void testExecutionWhenResultShouldBeCloseTo1() throws Exception {
        inputFriend.setValue(1.);
        inputVodka.setValue(1.);
        inputPartie.setValue(1.);
        assertTrue(outputNeuron.call() > .7);
    }

    @Test
    void testExecutionWhenResultShouldBeCloseTo0() throws Exception {
        inputFriend.setValue(0.);
        inputVodka.setValue(0.);
        inputPartie.setValue(0.);
        assertTrue(outputNeuron.call() < .3);
    }

    @Test
    void testExecutionWhenResultShouldBeMoreThen0dot5() throws Exception {
        inputFriend.setValue(1.);
        inputVodka.setValue(0.);
        inputPartie.setValue(1.);
        assertTrue(outputNeuron.call() > .5);
    }

    @Test
    void testExecutionWhenResultShouldBeLessThen0dot5() throws Exception {
        inputFriend.setValue(0.);
        inputVodka.setValue(0.);
        inputPartie.setValue(1.);
        assertTrue(outputNeuron.call() <= .5);
    }
}