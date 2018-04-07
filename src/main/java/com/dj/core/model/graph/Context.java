package com.dj.core.model.graph;

import java.io.Serializable;

public class Context implements Serializable {

    private double learningRate;

    private boolean debugMode;

    private int batchSize;

    public Context(final double learningRate, final boolean debugMode, int batchSize) {
        this.learningRate = learningRate;
        this.debugMode = debugMode;
        this.batchSize= batchSize;
    }

    public Context() {
        this(.2, false, 1);
    }


    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(final double learningRate) {
        this.learningRate = learningRate;
    }

    public boolean isDebugMode() {
        return debugMode;
    }

    public void setDebugMode(final boolean debugMode) {
        this.debugMode = debugMode;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
}
