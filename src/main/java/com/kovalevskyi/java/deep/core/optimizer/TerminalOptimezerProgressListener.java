package com.kovalevskyi.java.deep.core.optimizer;

public class TerminalOptimezerProgressListener implements OptimizerProgressListener {
    
    public void onProgress(
            final double loss,
            final int currentEpoch,
            final int targetEpoch) {
        System.out.printf(
                "LOSS: %5f, Epoch: %d of %d\n",
                loss,
                currentEpoch,
                targetEpoch);
    }
}
