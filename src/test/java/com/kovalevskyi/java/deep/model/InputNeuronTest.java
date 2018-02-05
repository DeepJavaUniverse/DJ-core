package com.kovalevskyi.java.deep.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class InputNeuronTest {
    
    @Test
    void callWhenValueIs1() throws Exception {
        InputNeuron neuron = new InputNeuron();
        neuron.setValue(1.);

        assertEquals((Double) 1., neuron.call());
    }
}