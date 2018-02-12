package com.kovalevskyi.java.deep.model.graph;

import com.kovalevskyi.java.deep.model.activation.ActivationFunction;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Function;

public class ConnectedNeuron implements Neuron {

    private final Map<Neuron, Double> backwardConnections;

    private final Set<Neuron> forwardConnections = new HashSet<>();

    private final double bias;

    private final ActivationFunction activationFunction;

    private volatile double forwardResult;

    private volatile boolean calculated;

    public ConnectedNeuron(final Map<Neuron, Double> backwardConnections,
                           final double bias,
                           final ActivationFunction activationFunction) {
        
        this.backwardConnections = backwardConnections;
        this.bias = bias;
        this.activationFunction = activationFunction;
    }

//    @Override
//    public Double call() throws Exception {
//        if (calculated){
//            return forwardResult;
//        }
//        double connectionsProcessingResult = backwardConnections
//                .entrySet()
//                .stream()
//                .map(connection -> Map.entry(FORK_JOIN_POOL.submit(connection.getKey()), connection.getValue()))
//                .map(task -> {
//                    try {
//                        return Map.entry(task.getKey().get(), task.getValue());
//                    } catch (Exception e) {
//                        throw new RuntimeException("Neuron processing exception", e);
//                    }
//                }).mapToDouble(res -> res.getKey() * res.getValue())
//                .sum() + bias;
//        forwardResult = activationFunction.forward(connectionsProcessingResult);
//        if (learning) {
//             backwardResult = activationFunction.backward(connectionsProcessingResult);
//        }
//        calculated = true;
//        return forwardResult;
//    }

    @Override
    public void invalidate() {
        calculated = false;
        backwardConnections.keySet().forEach(Neuron::invalidate);
    }

    @Override
    public Set<Neuron> backwardConnections() {
        return backwardConnections.keySet();
    }

    @Override
    public Boolean calculated() {
        return calculated;
    }

    @Override
    public Double forwardResult() {
        if (!calculated) {
            throw new RuntimeException("Not yet calculated");
        }
        return forwardResult;
    }

    @Override
    public Boolean canBeCalculatedForward() {
        return !backwardConnections.keySet().stream().filter(n -> !n.calculated()).findAny().isPresent();
    }

    @Override
    public Double calculateForward() {
        if (calculated) {
            return forwardResult;
        }
        if (backwardConnections
                .keySet()
                .stream()
                .map(Neuron::calculated)
                .<Boolean, Boolean>filter(calculated -> !calculated)
                .count() > 0) {
           throw new RuntimeException("Not all neurons are calculated!");
        }
        forwardResult = backwardConnections
                .entrySet()
                .stream()
                .mapToDouble(connection -> connection.getKey().forwardResult() * connection.getValue())
                .sum() + bias;
        forwardResult = activationFunction.forward(forwardResult);
        calculated = true;
        return forwardResult;
    }

    @Override
    public void addForwardConnection(final Neuron neuron) {
        forwardConnections.add(neuron);
    }

    @Override
    public void addBackwardConnection(final Neuron neuron, final Double weight) {
        backwardConnections.put(neuron, weight);
    }
}
