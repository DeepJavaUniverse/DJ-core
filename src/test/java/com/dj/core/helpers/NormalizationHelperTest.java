package com.dj.core.helpers;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

public class NormalizationHelperTest {

    @Test
    public void normalize() {
        final double[][] input = {{256., 128., 0}, {256., 0., 128.}};
        final double[][] expectedOutput = {{1., 0., -1.}, {1., -1., 0.}};

        final double[][] actualOutput = NormalizationHelper.normalize(input);

        assertTrue(Arrays.deepEquals(expectedOutput, actualOutput));
    }
}