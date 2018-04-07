package com.dj.core.model.visitor;

import org.apache.commons.math3.linear.RealMatrixChangingVisitor;

public abstract class SimpleRealMatrixChangingVisitor implements RealMatrixChangingVisitor {

    @Override
    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) { }

    @Override
    public double end() {
        return 0;
    }
}
