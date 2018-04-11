package com.dj.core.model.graph;

import java.io.Serializable;

public class Context implements Serializable {

    private double learningRate;

    private boolean debugMode;

    private int regularizationLevel = 2;

    private double regularizationRate = 0.;

    public Context(final double learningRate, final boolean debugMode) {
        this.learningRate = learningRate;
        this.debugMode = debugMode;
    }

    public Context() {
        this(.2, false);
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

    public int getRegularizationLevel() {
        return regularizationLevel;
    }

    public void setRegularizationLevel(final int regularizationLevel) {
        this.regularizationLevel = regularizationLevel;
    }

    public double getRegularizationRate() {
        return regularizationRate;
    }

    public void setRegularizationRate(final double regularizationRate) {
        this.regularizationRate = regularizationRate;
    }
}
