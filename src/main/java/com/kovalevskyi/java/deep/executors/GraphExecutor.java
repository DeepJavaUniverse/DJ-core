package com.kovalevskyi.java.deep.executors;

import com.kovalevskyi.java.deep.model.graph.Neuron;

import java.util.List;

public interface GraphExecutor<T> {

    List<T> execute(final List<Neuron> neurons);

}
