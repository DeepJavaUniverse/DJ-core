package com.kovalevskyi.java.deep.model;

import com.kovalevskyi.java.deep.executors.ForkJoinExecutor;
import com.kovalevskyi.java.deep.executors.GraphExecutor;
import com.kovalevskyi.java.deep.model.activation.Sigmoid;
import com.kovalevskyi.java.deep.model.graph.ConnectedNeuron;
import com.kovalevskyi.java.deep.model.graph.InputNeuron;
import com.kovalevskyi.java.deep.model.graph.Neuron;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SingleLayerPerceptronTest {

    private Neuron outputNeuron;
    private InputNeuron inputFriend;
    private InputNeuron inputVodka;
    private InputNeuron inputParty;
    private GraphExecutor<Double> graphExecutor;

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
        graphExecutor = new ForkJoinExecutor();
    }

    @Test
    void testExecutionAfterInvalidatingTheState() throws Exception {
         inputFriend.setValue(1.);
         inputVodka.setValue(1.);
         inputParty.setValue(1.);
         graphExecutor.execute(List.of(outputNeuron));
         assertTrue(outputNeuron.calculateForward() > .7);
         outputNeuron.forwardInvalidate();

         inputFriend.setValue(0.);
         inputVodka.setValue(0.);
         inputParty.setValue(0.);
         graphExecutor.execute(List.of(outputNeuron));
         assertTrue(outputNeuron.calculateForward() < .3);
    }

    @Test
    void testExecutionWhenResultShouldBeCloseTo1() throws Exception {
        inputFriend.setValue(1.);
        inputVodka.setValue(1.);
        inputParty.setValue(1.);
        graphExecutor.execute(List.of(outputNeuron));
        assertTrue(outputNeuron.calculateForward() > .7);
    }

    @Test
    void testExecutionWhenResultShouldBeCloseTo0() throws Exception {
        inputFriend.setValue(0.);
        inputVodka.setValue(0.);
        inputParty.setValue(0.);
        graphExecutor.execute(List.of(outputNeuron));
        assertTrue(outputNeuron.calculateForward() < .3);
    }

    @Test
    void testExecutionWhenResultShouldBeMoreThen0dot5() throws Exception {
        inputFriend.setValue(1.);
        inputVodka.setValue(0.);
        inputParty.setValue(1.);
        graphExecutor.execute(List.of(outputNeuron));
        assertTrue(outputNeuron.calculateForward() > .5);
    }

    @Test
    void testExecutionWhenResultShouldBeLessThen0dot5() throws Exception {
        inputFriend.setValue(0.);
        inputVodka.setValue(0.);
        inputParty.setValue(1.);
        graphExecutor.execute(List.of(outputNeuron));
        assertTrue(outputNeuron.calculateForward() <= .5);
    }
}