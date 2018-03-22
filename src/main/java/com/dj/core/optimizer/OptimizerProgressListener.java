package com.dj.core.optimizer;


public interface OptimizerProgressListener {

    void onProgress(
            final double loss,
            final int currentEpoch,
            final int targetEpoch);
}
