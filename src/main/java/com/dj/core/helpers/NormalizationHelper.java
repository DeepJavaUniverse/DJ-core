package com.dj.core.helpers;

import com.google.common.collect.Streams;

import java.util.Arrays;
import java.util.stream.IntStream;

public class NormalizationHelper {

    private NormalizationHelper() { }

    public static double[][] normalize(final double[][] input) {
        final double max = Arrays.stream(input).flatMapToDouble(row -> Arrays.stream(row)).max().getAsDouble();
        final double min = Arrays.stream(input).flatMapToDouble(row -> Arrays.stream(row)).min().getAsDouble();
        final double mid = min + (max - min) / 2.;
        final double[][] output = new double[input.length][];
        IntStream.range(0, input.length).forEach(i ->
            output[i] = Arrays.stream(input[i]).map(value -> (value - mid) / mid).toArray()
        );
        return output;
    }
}
