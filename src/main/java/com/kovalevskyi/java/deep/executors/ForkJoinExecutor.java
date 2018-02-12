package com.kovalevskyi.java.deep.executors;

import com.kovalevskyi.java.deep.model.graph.Neuron;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class ForkJoinExecutor implements GraphExecutor<Double> {

    private static final ForkJoinPool FORK_JOIN_POOL = new ForkJoinPool();

    @Override
    public List<Double> execute(final List<Neuron> neurons) {
        neurons.stream().map(ExecutionTask::new).forEach(FORK_JOIN_POOL::execute);
        return neurons.stream().map(Neuron::forwardResult).collect(Collectors.toList());
    }

    private class ExecutionTask extends RecursiveAction {

        private final Neuron neuronToCalculate;

        private ExecutionTask(final Neuron neuronToCalculate) {
            this.neuronToCalculate = neuronToCalculate;
        }

        @Override
        protected void compute() {
            if (neuronToCalculate.calculated()) {
                neuronToCalculate.forwardResult();
                return;
            }
            if (neuronToCalculate.canBeCalculatedForward()) {
                neuronToCalculate.calculateForward();
                return;
            }
            neuronToCalculate
                    .backwardConnections()
                    .stream()
                    .map(ExecutionTask::new)
                    .map(ExecutionTask::fork)
                    .forEach(ForkJoinTask::join);
        }
    }
}
